import os
import torch
import numpy as np
from scipy.signal import hilbert
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


def generate_augmented_inputs(x, y, args):
    aug_list = []
    aug_list.append(identity_aug_for_tta(x, y, args))
    x_c = np.transpose(x, (0, 2, 1))
    aug_list.append(data_noise_f_for_tta(x_c, y, args))
    aug_list.append(data_mult_f_for_tta(x_c, y, args, mult_mod=0.1))
    aug_list.append(data_mult_f_for_tta(x_c, y, args, mult_mod=-0.1))
    aug_list.append(data_mult_f_for_tta(x_c, y, args, mult_mod=-0.2))
    aug_list.append(freq_mod_f_for_tta(x_c, y, args, flag='high'))
    aug_list.append(freq_mod_f_for_tta(x_c, y, args, flag='low'))

    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=1))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=2))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=3))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=4))
    aug_list.append(sliding_window_augmentation_for_tta(x, y, args, no=5))

    return aug_list


# data: samples * size * n_channels
# size: int(freq * window_size)
def data_noise_f_for_tta(data, labels, args):
    new_data = []
    new_labels = []
    noise_mod_val = 2
    size = args.time_sample_num
    n_channels = args.chn

    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def data_mult_f_for_tta(data, labels, args, mult_mod=0.1):
    new_data = []
    new_labels = []
    mult_mod = mult_mod
    size = args.time_sample_num
    n_channels = args.chn

    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def data_neg_f_for_tta(data, labels, args):
    new_data = []
    new_labels = []
    size = args.time_sample_num
    n_channels = args.chn

    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = -1 * data[i]
            data_t = data_t - np.min(data_t)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def freq_mod_f_for_tta(data, labels, args, flag='low'):
    new_data = []
    new_labels = []
    freq_mod = 0.2
    size = args.time_sample_num
    n_channels = args.chn

    if flag=='low':
        for i in range(len(labels)):
            if labels[i] >= 0:
                low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
                new_data.append(low_shift)
                new_labels.append(labels[i])

    elif flag=='high':
        for i in range(len(labels)):
            if labels[i] >= 0:
                high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
                new_data.append(high_shift)
                new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)
    new_data_ar = np.transpose(new_data_ar, (0, 2, 1))

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    new_data_ar = new_data_ar[:, :, :eeg_length]

    new_data_ar = torch.tensor(new_data_ar, dtype=torch.float32)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    new_data_ar = new_data_ar.unsqueeze(1)
    new_data_ar = new_data_ar.cuda()
    new_labels = new_labels.cuda()

    return new_data_ar, new_labels


def freq_shift(x, f_shift, dt=1 / 250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = len(x)
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((padding_len - len_x, num_channels))
    with_padding = np.vstack((x, padding))
    hilb_T = hilbert(with_padding, axis=0)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * f_shift * dt * t)
    for i in range(num_channels):
        shifted_sig[:, i] = (hilb_T[:, i] * shift_func)[:len_x].real
    return shifted_sig


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def sliding_window_augmentation_for_tta(data, labels, args, no=1):
    chs = data.shape[1]
    augmented_data = []
    augmented_labels = []
    window_size = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    if args.data == 'BNCI2014001':
        stride = args.sample_rate * 0.2
    elif args.data == 'BNCI2014004':
        stride = args.sample_rate * 0.1
    elif args.data == 'Zhou2016':
        stride = args.sample_rate * 0.2
    elif args.data == 'Schirrmeister2017':
        stride = args.sample_rate * 0.2

    for i in range(len(data)):
        trial = data[i]  # shape: [C, T]
        label = labels[i]
        T = trial.shape[1]

        for start in range(0, int(T - window_size + 1), int(stride)):
            end = start + window_size
            window = trial[:, start:end]  # shape: [C, window_size]
            if start == int(stride) * no:
                augmented_data.append(window)
                augmented_labels.append(label)

    augmented_data = np.array(augmented_data).reshape([-1, chs, window_size])
    augmented_labels = np.array(augmented_labels)

    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    augmented_data = augmented_data[:, :, :eeg_length]

    augmented_data = torch.tensor(augmented_data, dtype=torch.float32)
    augmented_labels = torch.tensor(augmented_labels, dtype=torch.long)
    augmented_data = augmented_data.unsqueeze(1)
    augmented_data = augmented_data.cuda()
    augmented_labels = augmented_labels.cuda()

    return augmented_data, augmented_labels


def identity_aug_for_tta(data, labels, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    augmented_data = data[:, :, :eeg_length]

    augmented_data = torch.tensor(augmented_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    augmented_data = augmented_data.unsqueeze(1)
    augmented_data = augmented_data.cuda()
    labels = labels.cuda()

    return augmented_data, labels


def test_BNadapt(base_network, X_test, labels_test, args, test_batch=8):
    base_network = base_network.cuda()
    base_network.eval()

    X_test_c, labels_test_c = X_test, labels_test
    aug_dir = ['None']
    all_output = {}
    for aug_way in aug_dir:
        X_test, labels_test = X_test_c, labels_test_c
        X_test_trans = np.transpose(X_test, (0, 2, 1))
        if aug_way == 'None':
            X_test, labels_test = identity_aug_for_tta(X_test, labels_test, args)
        elif aug_way == 'mult_flag_0.9':
            X_test, labels_test = data_mult_f_for_tta(X_test_trans, labels_test, args, mult_mod=0.1)
        elif aug_way == 'mult_flag_1.1':
            X_test, labels_test = data_mult_f_for_tta(X_test_trans, labels_test, args, mult_mod=-0.1)
        elif aug_way == 'mult_flag_1.2':
            X_test, labels_test = data_mult_f_for_tta(X_test_trans, labels_test, args, mult_mod=-0.2)
        elif aug_way == 'noise_flag':
            X_test, labels_test = data_noise_f_for_tta(X_test_trans, labels_test, args)
        elif aug_way == 'high_freq_mod_flag':
            X_test, labels_test = freq_mod_f_for_tta(X_test_trans, labels_test, args, flag='high')
        elif aug_way == 'low_freq_mod_flag':
            X_test, labels_test = freq_mod_f_for_tta(X_test_trans, labels_test, args, flag='low')
        elif aug_way == 'slide_wds_flag_1':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=1)
        elif aug_way == 'slide_wds_flag_2':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=2)
        elif aug_way == 'slide_wds_flag_3':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=3)
        elif aug_way == 'slide_wds_flag_4':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=4)
        elif aug_way == 'slide_wds_flag_5':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=5)
        
        data_test = torch.utils.data.TensorDataset(X_test, labels_test)
        loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                              shuffle=False, drop_last=False)

        with torch.no_grad():
            iter_test = iter(loader_test)
            for i in range(len(loader_test)):
                base_network.eval()
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()

                if i == 0:    data_cum = inputs
                else:    data_cum = torch.cat((data_cum, inputs), 0)

                if i == 0:  all_label = labels.float()
                else:       all_label = torch.cat((all_label, labels.float()), 0)

                # print(inputs.shape)
                outputs = base_network(inputs)

                if i == 0:
                    all_output[aug_way] = outputs.float().cpu()                          
                else:
                    all_output[aug_way] = torch.cat((all_output[aug_way], outputs.float().cpu()), 0)

                if aug_way == 'None':
                    base_network.train()
                    if (i + 1) >= test_batch:
                        batch_test = data_cum[i - test_batch + 1: i + 1]
                        batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                        batch_test = batch_test.cuda()
                        _ = base_network(batch_test)

    for aug_way in aug_dir:
        this_outputs = all_output[aug_way]   
        this_outputs = nn.Softmax(dim=1)(this_outputs)
        
        _, predict = torch.max(this_outputs, 1)
        pred = torch.squeeze(predict).float()
        true = all_label.cpu()
        acc = accuracy_score(true, pred) * 100
        print('BNadapt: ' + 'Test Acc = {:.2f}'.format(acc))


def test_augment(base_network, X_test, labels_test, args, test_batch=8):
    base_network = base_network.cuda()
    base_network.eval()

    X_test_c, labels_test_c = X_test, labels_test
    aug_dir = ['None', 'mult_flag_0.9', 'mult_flag_1.1', 'mult_flag_1.2', 'noise_flag', 
               'high_freq_mod_flag', 'low_freq_mod_flag', 'slide_wds_flag_1', 
               'slide_wds_flag_2', 'slide_wds_flag_3', 'slide_wds_flag_4', 'slide_wds_flag_5']
    # aug_dir = ['None']
    all_output = {}
    for aug_way in aug_dir:
        X_test, labels_test = X_test_c, labels_test_c
        X_test_trans = np.transpose(X_test, (0, 2, 1))
        if aug_way == 'None':
            X_test, labels_test = identity_aug_for_tta(X_test, labels_test, args)
        elif aug_way == 'mult_flag_0.9':
            X_test, labels_test = data_mult_f_for_tta(X_test_trans, labels_test, args, mult_mod=0.1)
        elif aug_way == 'mult_flag_1.1':
            X_test, labels_test = data_mult_f_for_tta(X_test_trans, labels_test, args, mult_mod=-0.1)
        elif aug_way == 'mult_flag_1.2':
            X_test, labels_test = data_mult_f_for_tta(X_test_trans, labels_test, args, mult_mod=-0.2)
        elif aug_way == 'noise_flag':
            X_test, labels_test = data_noise_f_for_tta(X_test_trans, labels_test, args)
        elif aug_way == 'high_freq_mod_flag':
            X_test, labels_test = freq_mod_f_for_tta(X_test_trans, labels_test, args, flag='high')
        elif aug_way == 'low_freq_mod_flag':
            X_test, labels_test = freq_mod_f_for_tta(X_test_trans, labels_test, args, flag='low')
        elif aug_way == 'slide_wds_flag_1':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=1)
        elif aug_way == 'slide_wds_flag_2':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=2)
        elif aug_way == 'slide_wds_flag_3':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=3)
        elif aug_way == 'slide_wds_flag_4':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=4)
        elif aug_way == 'slide_wds_flag_5':
            X_test, labels_test = sliding_window_augmentation_for_tta(X_test, labels_test, args, no=5)
        
        data_test = torch.utils.data.TensorDataset(X_test, labels_test)
        loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                              shuffle=False, drop_last=False)

        with torch.no_grad():
            iter_test = iter(loader_test)
            for i in range(len(loader_test)):
                base_network.eval()
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()

                # if i == 0:    data_cum = inputs
                # else:    data_cum = torch.cat((data_cum, inputs), 0)

                if i == 0:  all_label = labels.float()
                else:       all_label = torch.cat((all_label, labels.float()), 0)

                # print(inputs.shape)
                outputs = base_network(inputs)

                if i == 0:
                    all_output[aug_way] = outputs.float().cpu()                          
                else:
                    all_output[aug_way] = torch.cat((all_output[aug_way], outputs.float().cpu()), 0)

                # if aug_way == 'None':
                #     base_network.train()
                #     if (i + 1) >= test_batch:
                #         batch_test = data_cum[i - test_batch + 1: i + 1]
                #         batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                #         batch_test = batch_test.cuda()
                #         _ = base_network(batch_test)

    for aug_way in aug_dir:
        this_outputs = all_output[aug_way]   
        this_outputs = nn.Softmax(dim=1)(this_outputs)

        if aug_way == aug_dir[1]:
            output_tensor = this_outputs.float().cpu().unsqueeze(0)
            # print(output_tensor.shape)
        elif aug_way != aug_dir[0]:
            output_tensor = torch.cat((output_tensor, this_outputs.float().cpu().unsqueeze(0)), dim=0)
            # print(output_tensor.shape)
        
        _, predict = torch.max(this_outputs, 1)
        pred = torch.squeeze(predict).float()
        true = all_label.cpu()
        acc = accuracy_score(true, pred) * 100
        print(aug_way + ': ' + 'Test Acc = {:.2f}'.format(acc))

    mean_output = output_tensor.mean(dim=0)
    # print(mean_output.shape)
    _, predict = torch.max(mean_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Augment avg test Acc = {:.2f}'.format(acc))


def test_augment_with_loss(model_loss, model_target, block_model, X_test, labels_test, args, test_batch=8):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)

    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                               shuffle=False, drop_last=False)
    
    model_loss.eval()
    iter_test = iter(loader_test)
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    with torch.no_grad():
        for i in range(len(loader_test)):
            model_target.eval()
            block_model.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]

            inputs_c = inputs[:, :, :eeg_length]
            inputs_c = inputs_c.unsqueeze(1)
            if i == 0:    data_cum = inputs_c
            else:    data_cum = torch.cat((data_cum, inputs_c), 0)
            
            inputs = inputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            x_aug_list = generate_augmented_inputs(inputs, labels, args)
            labels = torch.tensor(labels, dtype=torch.long)

            pred_losses = []
            for j in range(len(x_aug_list)):
                x, _ = x_aug_list[j]
                x = block_model(x)
                pred_losses.append(model_loss(x))

            pred_losses = torch.stack(pred_losses).squeeze()
            pred_losses = F.softmax(-pred_losses, dim=0)

            if i == 0:
                all_pred_losses = pred_losses.unsqueeze(0)
            else:
                all_pred_losses = torch.cat([all_pred_losses, pred_losses.unsqueeze(0)], dim=0)
            if i == 0:
                predicted_probs = all_pred_losses.mean(dim=0)
            else:
                predicted_probs = all_pred_losses.mean(dim=0)

            # top3_values, top3_indices = torch.topk(predicted_probs, k=5, largest=True)
            # the_output = []
            # for k in top3_indices:
            #     x, y = x_aug_list[k]
            #     target_output = model_target(x)
            #     target_output = target_output / 0.25
            #     # print(target_output.shape)
            #     the_output.append(nn.Softmax(dim=1)(target_output))
            # mean_output = torch.mean(torch.stack(the_output), dim=0)

            the_output = []
            for k in range(len(x_aug_list)):
                x, y = x_aug_list[k]
                target_output = model_target(x)
                target_output = target_output / 0.25
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)


            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

            model_target.train()
            block_model.train()
            if (i + 1) >= test_batch:
                batch_test = data_cum[i - test_batch + 1: i + 1]
                batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                batch_test = batch_test.cuda()
                _ = model_target(batch_test)
                _ = block_model(batch_test)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Aug with loss test Acc = {:.2f}'.format(acc))

