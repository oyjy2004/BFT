import sys
sys.path.append('/mnt/data2/oyjy/test-time/test-time-aug/regression_BFT/utils')
from augment import *
import argparse
from getdata import *
from EA import *


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, subject_ids, aug_names):
        """
        data_root: 增强数据保存的根目录（不包含 s0/s1...）
        subject_ids: 所有被试的编号列表，如 [0, 1, ..., 9]
        aug_names: 所有增强方法名，如 ['gaussian', 'permute', ...]
        """
        self.aug_names = aug_names
        self.samples = []  # [(x1, y1), (x2, y2), ...]

        for sid in subject_ids:
            subject_dir = os.path.join(data_root, f"s{sid}")
            aug_data = [torch.load(os.path.join(subject_dir, f"{aug}.pt"), map_location='cpu') for aug in aug_names]

            num_samples = aug_data[0]['data'].shape[0]  #
            for i in range(num_samples):
                x_aug_list = []
                for aug in aug_data:
                    x = aug['data'][i]
                    y = aug['labels'][i]
                    x_aug_list.append((x, y))
                self.samples.append(x_aug_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def save_aug_list(aug_list, save_dir, aug_names):
    os.makedirs(save_dir, exist_ok=True)
    for (data, labels), name in zip(aug_list, aug_names):
        save_path = os.path.join(save_dir, f"{name}.pt")
        torch.save({'data': data, 'labels': labels}, save_path)
        print(f"[✓] Saved {name} to {save_path}")


def load_aug(save_dir, name):
    path = os.path.join(save_dir, f"{name}.pt")
    data_dict = torch.load(path)
    return data_dict['data'], data_dict['labels']


if __name__ == '__main__':
    aug_names = [
        "identity", "noise",
        "mult_0.9", "mult_1.1", "mult_1.2",
        "freq_high", "freq_low",
        "slide_1", "slide_2", "slide_3", "slide_4", "slide_5"
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
    # data_name_list = ['Driving', 'New_driving', 'Seed']
    data_name_list = ['Seed']

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
        
        if args.data == 'Driving':
            eeg_path = "/mnt/data2/oyjy/Data/Driving/Driving_eeg_filter.pkl"
            label_path = "/mnt/data2/oyjy/Data/Driving/Driving_labels.pkl"
        elif args.data == 'New_driving':
            eeg_path = "/mnt/data2/oyjy/Data/New_driving/NewDri_eeg.pkl"
            label_path = "/mnt/data2/oyjy/Data/New_driving/NewDri_label.pkl"
        elif args.data == 'Seed':
            eeg_path = "/mnt/data2/oyjy/Data/SEED/SEED_eeg_f.pkl"
            label_path = "/mnt/data2/oyjy/Data/SEED/SEED_labels.pkl"

        EEG, LABEL = load_data(eeg_path, label_path, args)

        for i in range(len(EEG)):
            EEG[i] = EA_offline(EEG[i], 1)

        for i in range(len(EEG)):
            args.testID = i
            test_X, test_Y = get_testset(EEG, LABEL, args)
            
            aug_list = generate_augmented_inputs(test_X, test_Y, args)

            save_dir = '/mnt/data2/oyjy/test-time/test-time-aug/regression_BFT/augmented_data/'
            save_aug_list(aug_list, save_dir=save_dir + data_name + '/s' + str(i), aug_names=aug_names)

            # # data, label = load_aug("./augmented_data/seed_subject_0", "noise")
