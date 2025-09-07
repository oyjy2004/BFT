import torch
import time
from augment import *


def test_augment_with_loss_nq(model_loss, model_target, block_model,  X_test, labels_test, args, test_batch=8):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)

    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

    iter_test = iter(loader_test)
    model_loss.eval()
    augment_time = []
    forward_time = []
    with torch.no_grad():
        for i in range(len(loader_test)):
            block_model.eval()
            model_target.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]

            inputs_c = inputs[:, :, :eeg_length]
            inputs_c = inputs_c.unsqueeze(1)
            if i == 0:    data_cum = inputs_c
            else:    data_cum = torch.cat((data_cum, inputs_c), 0)
            
            ###################### augment ####################
            inputs = inputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            start_time = time.time()
            x_aug_list = generate_augmented_inputs(inputs, labels, args)
            end_time = time.time()
            augment_time.append(end_time - start_time)
            labels = torch.tensor(labels, dtype=torch.long)
            ###################### augment ####################
            ###################### forward ####################
            start_time = time.time()
            pred_losses = []
            for j in range(len(x_aug_list)):
                x, _ = x_aug_list[j]
                x = x.to(args.data_env)
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

            the_output = []
            for k in range(len(x_aug_list)):
                x, y = x_aug_list[k]
                x = x.to(args.data_env)
                target_output = model_target(x)
                target_output = target_output / 0.25
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)

            block_model.train()
            model_target.train()
            if (i + 1) >= test_batch:
                batch_test = data_cum[i - test_batch + 1: i + 1]
                batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                batch_test = batch_test.to(args.data_env)
                _ = model_target(batch_test)
                _ = block_model(batch_test)
                

            end_time = time.time()
            forward_time.append(end_time - start_time)
            ###################### forward ####################

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Aug with loss test Acc = {:.2f}'.format(acc))
    return acc, 1000 * np.mean(augment_time), 1000 * np.mean(forward_time)


def test_dropout_with_loss_nq(model_loss, block_model, classifier, X_test, labels_test, args, test_batch=8):
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.to(args.data_env), labels_test.to(args.data_env)
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)
    iter_test = iter(loader_test)

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]
    
    classifier.eval()
    model_loss.eval()
    all_time = []
    with torch.no_grad():
        for i in range(len(loader_test)):
            block_model.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(args.data_env)

            if i == 0:    data_cum = inputs
            else:    data_cum = torch.cat((data_cum, inputs), 0)

            ################ dropout & forward ##################
            start_time = time.time()
            pred_losses = []
            output1 = block_model(inputs)
            # print(output1)
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

            the_output = []
            for k in range(all_mask.shape[0]):
                x = all_mask[k]
                target_output = classifier(x)
                target_output = target_output / 0.25
                target_output = target_output.unsqueeze(0)
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)

            block_model.train()
            if (i + 1) >= test_batch:
                batch_test = data_cum[i - test_batch + 1: i + 1]
                batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
                batch_test = batch_test.to(args.data_env)
                _ = block_model(batch_test)
            end_time = time.time()
            all_time.append(end_time - start_time)
            ################ dropout & forward ##################

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Dropout with loss test Acc = {:.2f}'.format(acc))
    return acc, 0, 1000 * np.mean(all_time)


def test_nq(block_model, classifier, X_test, labels_test, args, test_batch=8):
    X_test, labels_test = identity_aug_for_tta(X_test, labels_test, args)
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

    block_model = block_model.to(args.data_env)
    classifier = classifier.to(args.data_env)
    block_model.eval()
    classifier.eval()
    all_time = []
    all_output = []
    with torch.no_grad():
        iter_test = iter(loader_test)
        for i in range(len(loader_test)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(args.data_env)

            if i == 0:    data_cum = inputs
            else:         data_cum = torch.cat((data_cum, inputs), 0)

            if i == 0:  all_label = labels.float()
            else:       all_label = torch.cat((all_label, labels.float()), 0)

            start_time = time.time()
            outputs = classifier(block_model(inputs))
            end_time = time.time()
            all_time.append(end_time - start_time)

            if i == 0:
                all_output = outputs.float().cpu()                          
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Test Acc = {:.2f}'.format(acc))
    return acc, 0, 1000 * np.mean(all_time)


def test_q(block_model, classifier, X_test, labels_test, args, test_batch=8):
    X_test, labels_test = identity_aug_for_tta(X_test, labels_test, args)
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

    block_model = block_model.cpu()
    classifier = classifier.cpu()
    block_model.eval()
    classifier.eval()
    all_time = []
    all_output = []
    with torch.no_grad():
        iter_test = iter(loader_test)
        for i in range(len(loader_test)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cpu()

            if i == 0:    data_cum = inputs
            else:         data_cum = torch.cat((data_cum, inputs), 0)

            if i == 0:  all_label = labels.float()
            else:       all_label = torch.cat((all_label, labels.float()), 0)

            start_time = time.time()
            outputs = classifier(block_model(inputs))
            end_time = time.time()
            all_time.append(end_time - start_time)

            if i == 0:
                all_output = outputs.float().cpu()                          
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Test Acc = {:.2f}'.format(acc))
    return acc, 0, 1000 * np.mean(all_time)


def test_augment_with_loss_q(model_loss, block_model, classifier, X_test, labels_test, args, test_batch=8):
    block_model = block_model.cpu()
    model_loss = model_loss.cpu()
    classifier = classifier.cpu()
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)

    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)

    iter_test = iter(loader_test)
    model_loss.eval()
    augment_time = []
    forward_time = []
    with torch.no_grad():
        for i in range(len(loader_test)):
            block_model.eval()
            classifier.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]

            inputs_c = inputs[:, :, :eeg_length]
            inputs_c = inputs_c.unsqueeze(1)
            if i == 0:    data_cum = inputs_c
            else:    data_cum = torch.cat((data_cum, inputs_c), 0)
            
            ###################### augment ####################
            inputs = inputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            start_time = time.time()
            x_aug_list = generate_augmented_inputs(inputs, labels, args)
            end_time = time.time()
            augment_time.append(end_time - start_time)
            labels = torch.tensor(labels, dtype=torch.long)
            ###################### augment ####################
            ###################### forward ####################
            start_time = time.time()
            pred_losses = []
            for j in range(len(x_aug_list)):
                x, _ = x_aug_list[j]
                x = x.cpu()
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

            the_output = []
            for k in range(len(x_aug_list)):
                x, y = x_aug_list[k]
                x = x.cpu()
                target_output = classifier(block_model(x))
                target_output = target_output / 0.25
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)

            # block_model.train()
            # if (i + 1) >= test_batch:
            #     batch_test = data_cum[i - test_batch + 1: i + 1]
            #     batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
            #     batch_test = batch_test.cpu()
            #     _ = block_model(batch_test)

            end_time = time.time()
            forward_time.append(end_time - start_time)
            ###################### forward ####################

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Aug with loss test Acc = {:.2f}'.format(acc))
    return acc, 1000 * np.mean(augment_time), 1000 * np.mean(forward_time)


def test_dropout_with_loss_q(model_loss, block_model, classifier, X_test, labels_test, args, test_batch=8):
    block_model = block_model.cpu()
    model_loss = model_loss.cpu()
    classifier = classifier.cpu()
    eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
    X_test = X_test[:, :, :eeg_length]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.long)
    X_test = X_test.unsqueeze(1)
    X_test, labels_test = X_test.cpu(), labels_test.cpu()
    
    data_test = torch.utils.data.TensorDataset(X_test, labels_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=False)
    iter_test = iter(loader_test)

    num_splits = args.dropout_num
    drop_ranges = [(i / num_splits, (i + 1) / num_splits) for i in range(num_splits)]
    range_keys = [f"{start:.1f}-{end:.1f}" for start, end in drop_ranges]
    
    classifier.eval()
    model_loss.eval()
    all_time = []
    with torch.no_grad():
        for i in range(len(loader_test)):
            block_model.eval()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cpu()

            if i == 0:    data_cum = inputs
            else:    data_cum = torch.cat((data_cum, inputs), 0)

            ################ dropout & forward ##################
            start_time = time.time()
            pred_losses = []
            output1 = block_model(inputs)
            # print(output1)
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

            the_output = []
            for k in range(all_mask.shape[0]):
                x = all_mask[k]
                target_output = classifier(x)
                target_output = target_output / 0.25
                target_output = target_output.unsqueeze(0)
                the_output.append(nn.Softmax(dim=1)(target_output))
            mean_output = (torch.stack(the_output).squeeze(1) * predicted_probs.unsqueeze(1)) / predicted_probs.sum()
            mean_output = mean_output.sum(dim=0).unsqueeze(0)

            # block_model.train()
            # if (i + 1) >= test_batch:
            #     batch_test = data_cum[i - test_batch + 1: i + 1]
            #     batch_test = batch_test.reshape(test_batch, 1, batch_test.shape[2], batch_test.shape[3])
            #     batch_test = batch_test.cpu()
            #     _ = block_model(batch_test)

            end_time = time.time()
            all_time.append(end_time - start_time)
            ################ dropout & forward ##################

            if i == 0:
                all_output = mean_output.float().cpu()
                all_label = labels.float()
            else:
                all_output = torch.cat((all_output, mean_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred) * 100
    print('Dropout with loss test Acc = {:.2f}'.format(acc))
    return acc, 0, 1000 * np.mean(all_time)


def model_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters(recurse=True):
        if p is None:
            continue
        total += p.numel() * p.element_size()
    for b in model.buffers(recurse=True):
        if b is None:
            continue
        total += b.numel() * b.element_size()
    return total


def print_model_parameters(model, print_grad=False, print_size=False):
    """
    打印模型的所有参数和缓冲区信息
    
    参数:
        model: PyTorch模型
        print_grad: 是否打印梯度信息
        print_size: 是否打印每个参数的内存占用
    """
    print("=" * 80)
    print(f"模型结构: {model.__class__.__name__}")
    print("=" * 80)
    
    total_params = 0
    total_size = 0
    
    print("\n可训练参数:")
    print("-" * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            param_size = num_params * param.element_size()
            total_params += num_params
            total_size += param_size
            
            print(f"{name:40} | 形状: {str(list(param.shape)):20} | 类型: {param.dtype} | 参数数量: {num_params:8,}")
            if print_grad:
                print(f"{'':40} | 梯度: {param.grad is not None}")
            if print_size:
                print(f"{'':40} | 大小: {param_size:8,} bytes ({param_size/1024:.2f} KB)")
    
    print("\n缓冲区（不可训练参数）:")
    print("-" * 80)
    for name, buffer in model.named_buffers():
        num_params = buffer.numel()
        buffer_size = num_params * buffer.element_size()
        total_size += buffer_size
        
        print(f"{name:40} | 形状: {str(list(buffer.shape)):20} | 类型: {buffer.dtype} | 参数数量: {num_params:8,}")
        if print_size:
            print(f"{'':40} | 大小: {buffer_size:8,} bytes ({buffer_size/1024:.2f} KB)")
    
    print("\n汇总信息:")
    print("-" * 80)
    print(f"总参数量: {total_params:,}")
    print(f"总内存占用: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    if total_params != 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
