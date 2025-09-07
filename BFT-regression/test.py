import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import argparse
from utils.EA import *
from utils.getdata import *
from utils.fix_seed import *

from models.Deformer import *
from models.EEGNet import *
from models.lossPredictor import *
from augment_about import *
from dropout_about import *


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
    data_name_list = ['Driving', 'Seed']

    for data_name in data_name_list:
        if data_name == 'Driving': 
            paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'Ecog', 15, 30, 2000, 250, 512
        if data_name == 'New_driving': 
            paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'Ecog', 27, 30, 750, 250, 512
        if data_name == 'Seed': 
            paradigm, N, chn, time_sample_num, sample_rate, feature_deep_dim = 'Ecog', 23, 17, 1600, 200, 512

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn,  paradigm=paradigm, data=data_name)

        args.method = 'EEGNet'
        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True
        args.dropout_num = 10

        # cpu or cuda
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'

        # # get data
        # if args.data == 'Driving':
        #     eeg_path = "/mnt/data2/oyjy/Data/Driving/Driving_eeg_filter.pkl"
        #     label_path = "/mnt/data2/oyjy/Data/Driving/Driving_labels.pkl"
        # elif args.data == 'New_driving':
        #     eeg_path = "/mnt/data2/oyjy/Data/New_driving/NewDri_eeg.pkl"
        #     label_path = "/mnt/data2/oyjy/Data/New_driving/NewDri_label.pkl"
        # elif args.data == 'Seed':
        #     eeg_path = "/mnt/data2/oyjy/Data/SEED/SEED_eeg_f.pkl"
        #     label_path = "/mnt/data2/oyjy/Data/SEED/SEED_labels.pkl"

        # EEG, LABEL = load_data(eeg_path, label_path, args)

        # for i in range(len(EEG)):
        #     EEG[i] = EA_offline(EEG[i], 1)

        for SEED in [42, 43, 44]:
            args.SEED = SEED
            fix_random_seed(SEED)
            for testID in range(N):
                args.testID = testID
                # tar_data, tar_label = get_testset(EEG, LABEL, args)

                load_dir = "/mnt/data2/oyjy/test-time/test-time-aug/regression_BFT/data_noise/noise4/" + data_name + '/s' + str(args.testID) + '/dataset.pt'
                checkpoint = torch.load(load_dir)
                tar_data = checkpoint['data']     
                tar_label = checkpoint['labels']  
                tar_data = tar_data.detach().cpu().numpy()
                tar_label = tar_label.detach().cpu().numpy()
                tar_data = EA_offline(tar_data, 1)

                print(args.data, '  s' + str(testID))
                print("Target Data Shape: ", tar_data.shape, "Target Label Shape: ", tar_data.shape)

                eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
                EEGNet_model = EEGNet(Chans=args.chn,
                                Samples=eeg_length,
                                kernLength=int(args.sample_rate // 2),
                                F1=8,
                                D=2,
                                F2=16,
                                dropoutRate=0.25)  
                base_model = EEGNet_Block(EEGNet_model.block1, EEGNet_model.block2)
                regression_model = EEGNet_Regression(EEGNet_model.regression_block) 

                base_dir = '/mnt/data2/oyjy/test-time/test-time-aug/regression_BFT/checkpoints/SEED' + str(SEED) + '/EEGNet/New_EEGNetBlock/'
                if args.data  == 'Driving':
                    if args.testID == 5:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_25400.pth'
                    elif args.testID in [6, 10]:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_25300.pth'
                    elif args.testID == 12:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_25500.pth'
                    else:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_25200.pth'
                elif args.data  == 'Seed':
                    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/EEGNetBlock_epoch_30400.pth'
                tar_model_dir_cc = tar_model_dir
                checkpoint = torch.load(tar_model_dir)
                base_model.load_state_dict(checkpoint)
                base_model = base_model.cuda()

                base_dir = '/mnt/data2/oyjy/test-time/test-time-aug/regression_BFT/checkpoints/SEED' + str(SEED) + '/EEGNet/New_Regression_head/'
                if args.data  == 'Driving':
                    if args.testID == 5:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_25400.pth'
                    elif args.testID in [6, 10]:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_25300.pth'
                    elif args.testID == 12:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_25500.pth'
                    else:
                        tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_25200.pth'
                elif args.data  == 'Seed':
                    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/Regression_head_epoch_30400.pth'
                checkpoint = torch.load(tar_model_dir)
                regression_model.load_state_dict(checkpoint)
                regression_model = regression_model.cuda()

                test_augment(base_model, regression_model, tar_data, tar_label, args)
                test_BNadapt(base_model, regression_model, tar_data, tar_label, args)
                checkpoint = torch.load(tar_model_dir_cc)
                base_model.load_state_dict(checkpoint)
                base_model = base_model.cuda()
                test_dropout(base_model, regression_model, tar_data, tar_label, args)


                F2 = 16
                input_dim = F2 * (eeg_length // (4 * 8))
                model_loss = LossPredictor(input_dim)

                base_dir = '/mnt/data2/oyjy/test-time/test-time-aug/regression_BFT/checkpoints/SEED' + str(SEED) + '/EEGNet/New_loss_model_dropout/'
                if args.data  == 'Driving':
                    if args.testID == 5:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_10160.pth'
                    elif args.testID == 6:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_10120.pth'
                    elif args.testID == 12:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_10200.pth'
                    elif args.testID in [9, 11, 14]:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_10100.pth'
                    elif args.testID == 10:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_10140.pth'
                    else:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_10080.pth'
                elif data_name == 'Seed': 
                    loss_model_dir = base_dir + args.data + '/' + 's' + str(args.testID) + '/LossPredictor_epoch_12160.pth'

                checkpoint = torch.load(loss_model_dir)
                model_loss.load_state_dict(checkpoint)
                model_loss = model_loss.cuda()

                test_augment_with_loss(model_loss, base_model, regression_model, tar_data, tar_label, args)

                checkpoint = torch.load(tar_model_dir_cc)
                base_model.load_state_dict(checkpoint)
                base_model = base_model.cuda()
                test_dropout(base_model, regression_model, tar_data, tar_label, args)
                test_dropout_with_loss(model_loss, base_model, regression_model, tar_data, tar_label, args)