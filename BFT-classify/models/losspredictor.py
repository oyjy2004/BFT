import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from EEGNet import EEGNet, EEGNet_Block, EEGNet_Classifier, fix_random_seed
import sys
sys.path.append("..") 
from augment import *

import numpy as np
import sys

PATH_TO_SODDEP = '../sodeep'
sys.path.append(PATH_TO_SODDEP)
from sodeep import load_sorter, SpearmanLoss

    
class EEGNetLossPredictor(nn.Module):
    def __init__(self, F2, Samples):
        super().__init__()
        self.F2 = F2
        self.Samples = Samples
        self.losspre_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=(self.F2 * (self.Samples // (4 * 8))) // 2,
                    bias=True),
            nn.ELU(),
            nn.Linear(in_features=(self.F2 * (self.Samples // (4 * 8))) // 2,
                    out_features=(self.F2 * (self.Samples // (4 * 8))) // 4,
                    bias=True),  
            nn.ELU(),      
            nn.Linear(in_features=(self.F2 * (self.Samples // (4 * 8))) // 4,
                    out_features=1,
                    bias=True))

    def forward(self, x):
        output = self.losspre_block(x)
        return output


def compute_real_losses(augmented_inputs, model_target):
    loss_fn = nn.CrossEntropyLoss()
    real_losses = []
    model_target.eval()
    with torch.no_grad():
        for x_aug, label in augmented_inputs:
            pred = model_target(x_aug)
            loss = loss_fn(pred, label)
            real_losses.append(loss.item())
    return torch.tensor(real_losses)  # dim = 12


PATH_TO_LOSSPRE_MODEL = "/PATH/TO/SAVE/MODEL/"
PATH_TO_LOSSPRE_MODEL_DROPOUT = "/PATH/TO/SAVE/MODEL/"
def learn_augment_loss(model_loss, model_target, block_model, X_train, labels_train, args):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    sorter_checkpoint_path = PATH_TO_SODDEP + '/weights/12th_100epochs_best_model.pth.tar'
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    criterion.cuda()
    optimizer = optim.Adam(model_loss.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    interval_iter = int(args.max_epoch / 10) * max_iter // args.max_epoch
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    model_target.eval()
    block_model.eval()
    model_loss.train()
    while iter_num < max_iter:
        try:
            inputs_train, labels_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train, labels_train = next(iter_train)

        if inputs_train.size(0) == 1:
            continue

        iter_num += 1

        inputs_train = inputs_train.detach().cpu().numpy()
        labels_train = labels_train.detach().cpu().numpy()

        x_aug_list = generate_augmented_inputs(inputs_train, labels_train, args)

        real_losses = compute_real_losses(x_aug_list, model_target)
        # print(real_losses.shape)
        relative_real_losses = F.softmax(-real_losses, dim=0)
        relative_real_losses = relative_real_losses.cuda()

        pred_losses = []
        for i in range(len(x_aug_list)):
            x, _ = x_aug_list[i]
            x = block_model(x)
            pred_losses.append(model_loss(x))
        pred_losses = torch.stack(pred_losses).squeeze()
        # print(pred_losses.shape)
        pred_losses = pred_losses.mean(dim=1)   
        # print(pred_losses.shape)
        predicted_probs = F.softmax(-pred_losses, dim=0)

        loss = criterion(predicted_probs, relative_real_losses)
        epoch_loss += loss.item()
        cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            epoch_loss_avg = epoch_loss / cnt
            print('Epoch:{}/{}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            CHECKPOINT_DIR = PATH_TO_LOSSPRE_MODEL + str(args.SEED) + "/loss_model_new(batch=16)/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.idt) + "/loss_pre"
            if os.path.isdir(path):
                 pass
            else:
                 os.makedirs(path)
            torch.save(model_loss.state_dict(), 
                       path + "/EEGNetLossPredictor_epoch_" + str(iter_num) + ".pth")
            

def learn_dropout_loss(model_loss, block_model, classifier, X_train, labels_train, args):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_train = X_train[:, :, :eeg_length]
    X_train = X_train.unsqueeze(1)
    X_train, labels_train = X_train.cuda(), labels_train.cuda()

    data_train = torch.utils.data.TensorDataset(X_train, labels_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True)
    
    sorter_checkpoint_path = PATH_TO_SODDEP + 'weights/10th_100epochs_best_model.pth.tar'
    loss_fn = nn.CrossEntropyLoss()
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    criterion.cuda()
    optimizer = optim.Adam(model_loss.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(loader_train)
    interval_iter = int(args.max_epoch / 10) * max_iter // args.max_epoch
    iter_num = 0
    epoch_loss = 0
    cnt = 0

    block_model.eval()
    classifier.eval()
    model_loss.train()

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]

    while iter_num < max_iter:
        try:
            inputs_train, labels_train = next(iter_train)
        except:
            iter_train = iter(loader_train)
            inputs_train, labels_train = next(iter_train)

        if inputs_train.size(0) == 1:
            continue

        iter_num += 1

        pred_losses = []
        dropout_loss_list = []
        output1 = block_model(inputs_train)
        B, D = output1.shape
        for (start_r, end_r), key in zip(drop_ranges, range_keys):
            output1_mask = output1.clone()
            start = int(start_r * D)
            end = int(end_r * D)
            output1_mask[:, start:end] = 0.0

            # B * 2
            this_outputs = classifier(output1_mask)
            this_loss = loss_fn(this_outputs, labels_train)
            dropout_loss_list.append(this_loss.item())

            pred_losses.append(model_loss(output1_mask))
        # dim = 10
        real_losses = torch.tensor(dropout_loss_list)
        pred_losses = torch.stack(pred_losses).squeeze()
        # print(pred_losses.shape)
        pred_losses = pred_losses.mean(dim=1)   
        # print(real_losses.shape)
        # print(pred_losses.shape)
        relative_real_losses = F.softmax(-real_losses, dim=0)
        predicted_probs = F.softmax(-pred_losses, dim=0)
        relative_real_losses = relative_real_losses.cuda()

        loss = criterion(predicted_probs, relative_real_losses)
        epoch_loss += loss.item()
        cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            epoch_loss_avg = epoch_loss / cnt
            print('Epoch:{}/{}; Epoch Loss = {:.2f}'
                  .format(int(iter_num // len(loader_train)), 
                          int(max_iter // len(loader_train)), 
                          epoch_loss_avg))

            CHECKPOINT_DIR = PATH_TO_LOSSPRE_MODEL_DROPOUT + str(args.SEED) + "/loss_model_dropout_new(batch=16)/"
            path = CHECKPOINT_DIR + args.data + "/s" + str(args.idt) + "/loss_pre"
            if os.path.isdir(path):
                 pass
            else:
                 os.makedirs(path)
            torch.save(model_loss.state_dict(), 
                       path + "/EEGNetLossPredictor_epoch_" + str(iter_num) + ".pth")


if __name__ == '__main__':
    sorter_checkpoint_path = PATH_TO_SODDEP + '/weights/12th_50epochs_best_model.pth.tar'
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    a = [2, 3, 3.2, 1.5, 3.4, 6.7, 9, 1.1, 12.1, 11, 57, 100]
    b = [6.7, 9, 1.1, 12.1, 11, 57, 2, 3, 3.2, 1.5, 3.4, 100]
    a = torch.tensor(a)
    b = torch.tensor(b)
    print(a.shape)
    loss = criterion(a, b)
    print(loss)
    