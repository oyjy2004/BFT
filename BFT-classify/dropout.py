import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def test_dropout(block_model, classifier, X_test, labels_test, args):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    # [trials, 1, channels, samples]
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.cuda(), labels_test.cuda()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                            shuffle=False, drop_last=False)
    
    block_model.eval()
    classifier.eval()

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]
    all_output = {}
    
    with torch.no_grad():
        iter_test = iter(loader_test)
        for i in range(len(loader_test)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            if i == 0:  all_label = labels.float()
            else:       all_label = torch.cat((all_label, labels.float()), 0)

            output1 = block_model(inputs)
            B, D = output1.shape
            for (start_r, end_r), key in zip(drop_ranges, range_keys):
                output1_mask = output1.clone()
                start = int(start_r * D)
                end = int(end_r * D)
                output1_mask[:, start:end] = 0.0

                outputs = classifier(output1_mask)

                if i == 0:
                    all_output[key] = outputs.float().cpu()                          
                else:
                    all_output[key] = torch.cat((all_output[key], outputs.float().cpu()), 0)
                    
    for key, value_list in all_output.items():
        value_list = nn.Softmax(dim=1)(value_list)
        if key == range_keys[0]:
            output_tensor = value_list.float().cpu().unsqueeze(0)
            # print(output_tensor.shape)
        else:
            output_tensor = torch.cat((output_tensor, value_list.float().cpu().unsqueeze(0)), dim=0)
            # print(output_tensor.shape)

    mean_output = output_tensor.mean(dim=0)
    # print(mean_output.shape)
    _, predict = torch.max(mean_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Dropout avg test Acc = {:.2f}'.format(acc))


def test_dropout_with_loss(model_loss, block_model, classifier, X_test, labels_test, args, test_batch=8):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    # [trials, 1, channels, samples]
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.cuda(), labels_test.cuda()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                            shuffle=False, drop_last=False)
    
    classifier.eval()
    model_loss.eval()

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]

    iter_test = iter(loader_test)
    with torch.no_grad():
        for i in range(len(loader_test)):
            block_model.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            if i == 0:    data_cum = inputs
            else:    data_cum = torch.cat((data_cum, inputs), 0)

            pred_losses = []
            output1 = block_model(inputs)
            B, D = output1.shape
            all_mask = []
            for (start_r, end_r), key in zip(drop_ranges, range_keys):
                output1_mask = output1.clone()
                start = int(start_r * D)
                end = int(end_r * D)
                output1_mask[:, start:end] = 0.0
                all_mask.append(output1_mask)

                pred_losses.append(model_loss(output1_mask))
            # dim = 10 
            pred_losses = torch.stack(pred_losses).squeeze()
            pred_losses = F.softmax(-pred_losses, dim=0)

            # pred_losses = F.softmax(pred_losses, dim=0)

            # 10 * dim
            all_mask = torch.stack(all_mask).squeeze()

            if i == 0:
                all_pred_losses = pred_losses.unsqueeze(0)
            else:
                all_pred_losses = torch.cat([all_pred_losses, pred_losses.unsqueeze(0)], dim=0)
            if i == 0:
                predicted_probs = all_pred_losses.mean(dim=0)
            else:
                predicted_probs = all_pred_losses.mean(dim=0)
            # predicted_probs = pred_losses

            # top3_values, top3_indices = torch.topk(predicted_probs, k=5, largest=True)
            # the_output = []
            # for k in top3_indices:
            #     x = all_mask[k]
            #     target_output = classifier(x)
            #     target_output = target_output.unsqueeze(0)
            #     # print(target_output.shape)
            #     the_output.append(nn.Softmax(dim=1)(target_output))
            # mean_output = torch.mean(torch.stack(the_output), dim=0)
            the_output = []
            for k in range(all_mask.shape[0]):
                x = all_mask[k]
                target_output = classifier(x)
                target_output = target_output / 0.25
                target_output = target_output.unsqueeze(0)
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

            block_model.train()
            if (i + 1) >= test_batch:
                batch_test = data_cum[i - test_batch + 1: i + 1]
                batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                batch_test = batch_test.cuda()
                _ = block_model(batch_test)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Dropout with loss test Acc = {:.2f}'.format(acc))
    
    
