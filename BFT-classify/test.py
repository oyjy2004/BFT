import os 
import torch
import torch.nn as nn
from models.EEGNet import *
from models.losspredictor import *

import argparse
from utils.EA import *
from dropout import *
from augment import *
from utils.data_utils import *


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'
    data_name_list = ['Zhou2016', 'Schirrmeister2017']

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
            'MI', 9, 22, 2, 1001, 250, [144, 144, 144, 144, 144, 144, 144, 144, 144]
        if data_name == 'Zhou2016': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
            'MI', 4, 14, 2, 1251, 250, [119, 100, 100, 90]
        if data_name == 'Schirrmeister2017': 
            paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = \
            'MI', 14, 128, 2, 2001, 500, [160, 406, 440, 448, 360, 440, 440, 327, 441, 440, 440, 440, 400, 440]

        args = argparse.Namespace(trial_num=trial_num, time_sample_num=time_sample_num, 
                                  sample_rate=sample_rate, N=N, chn=chn, 
                                  class_num=class_num, paradigm=paradigm, data=data_name)

        args.method = 'EEGNet'
        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True
        args.dropout_num = 10

        # cpu or cuda
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'

        # load data
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)
        data_subjects, labels_subjects = split_data_by_subject(X, y, args.trial_num)
        
        # EA
        for i in range(len(data_subjects)):
            data_subjects[i] = EA_offline(data_subjects[i], 1)

        for SEED in [42, 43, 44]:
            args.SEED = SEED
            fix_random_seed(SEED)
            for idt in range(N):
                # target subject
                args.idt = idt
                src_data, src_label, tar_data, tar_label = get_test_train(data_subjects, labels_subjects, idt)

                load_dir = "/mnt/data2/oyjy/test-time/test-time-aug/classify_BFT/data_noise/noise3/" + data_name + '/s' + str(args.idt) + '/dataset.pt'
                checkpoint = torch.load(load_dir)
                tar_data = checkpoint['data']     
                tar_label = checkpoint['labels']  
                tar_data = tar_data.detach().cpu().numpy()
                tar_label = tar_label.detach().cpu().numpy()
                tar_data = EA_offline(tar_data, 1)
                
                eeg_length = (round(args.time_sample_num/args.sample_rate) - 1) * args.sample_rate
                model_target = EEGNet(n_classes=args.class_num,
                                Chans=args.chn,
                                Samples=eeg_length,
                                kernLength=int(args.sample_rate // 2),
                                F1=8,
                                D=2,
                                F2=16,
                                dropoutRate=0.25)   
                base_dir = '/mnt/data2/oyjy/test-time/test-time-aug/classify_BFT/checkpoints/EEGNet/SEED' + str(SEED) + '/EEGNet_pth/'
                if args.data  == 'BNCI2014001':
                    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_3600.pth'
                elif data_name == 'Zhou2016': 
                    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_1800.pth'
                elif data_name == 'Schirrmeister2017':
                    if idt in [0, ]:    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_34000.pth'
                    elif idt in [1, 12]:    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_32600.pth'
                    elif idt in [2, 3, 5, 6, 8, 9, 10, 11, 13]:    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_32200.pth'
                    elif idt in [4, ]:    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_32800.pth'
                    elif idt in [7, ]:    tar_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/EEGNet_epoch_33000.pth'
                tar_model_dir_cc = tar_model_dir
                checkpoint = torch.load(tar_model_dir)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()
                
                test_augment(model_target, tar_data, tar_label, args)
                test_BNadapt(model_target, tar_data, tar_label, args)
                checkpoint = torch.load(tar_model_dir_cc)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()

                model_loss = EEGNetLossPredictor(F2=16, Samples=eeg_length) 
                base_dir = '/mnt/data2/oyjy/test-time/test-time-aug/classify_BFT/checkpoints/EEGNet/SEED' + str(SEED) + '/loss_model_dropout_new(batch=16)/'
                if args.data  == 'BNCI2014001':
                    loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_1440.pth'
                elif data_name == 'Zhou2016': 
                    if idt == 0:    loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_720.pth'
                    else:   loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_760.pth'
                elif data_name == 'Schirrmeister2017': 
                    if idt in [0, ]:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_6820.pth'
                    elif idt in [1, 12]:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_6520.pth'
                    elif idt in [2, 3, 5, 6, 8, 9, 10, 11, 13]:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_6460.pth'
                    elif idt in [4, ]:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_6560.pth'
                    elif idt in [7, ]:
                        loss_model_dir = base_dir + args.data + '/' + 's' + str(args.idt) + '/loss_pre/EEGNetLossPredictor_epoch_6600.pth'
                checkpoint = torch.load(loss_model_dir)
                model_loss.load_state_dict(checkpoint)
                model_loss = model_loss.cuda()

                test_augment_with_loss(model_loss, model_target, block_model, tar_data, tar_label, args)
                checkpoint = torch.load(tar_model_dir_cc)
                model_target.load_state_dict(checkpoint)
                model_target = model_target.cuda()

                block_model = EEGNet_Block(model_target.block1, model_target.block2)
                classifier = EEGNet_Classifier(model_target.classifier_block)
                block_model = block_model.cuda()
                classifier = classifier.cuda()
                test_dropout(block_model, classifier, tar_data, tar_label, args)
                test_dropout_with_loss(model_loss, block_model, classifier, tar_data, tar_label, args)