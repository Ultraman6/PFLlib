import numpy as np
import os
import torch
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder

# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# Allocate data to users
def _generate_dataset(args, transform, partitioner):
    dir_path = os.path.join(args.data_root, 'TinyImagenet')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, args.num_clients, args.niid, args.balance, args.partition):
        return

    # Get data
    if not os.path.exists(f'{dir_path}/rawdata/'):
        os.system(f'wget --directory-prefix {dir_path}/rawdata/ http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system(f'unzip {dir_path}/rawdata/tiny-imagenet-200.zip -d {dir_path}/rawdata/')
    else:
        print('rawdata already exists.\n')

    trainset = ImageFolder_custom(root=dir_path+'/rawdata/tiny-imagenet-200/train/', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), args.num_clients, num_classes,
                                    args.niid, args.alpha, args.balance, args.partition)
    train_data, test_data = split_data(X, y, args.train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, args.num_clients, num_classes,
        statistic, args.niid, args.balance, args.partition)
