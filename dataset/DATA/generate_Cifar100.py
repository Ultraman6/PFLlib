import argparse
import numpy as np
import os
import torch
import torchvision
from dataset.utils.long_tailed import get_long_tail
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file


# Allocate data to users
def _generate_dataset(args, transform, partitioner):
    dir_path = os.path.join(args.data_root, 'CIFAR100')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, args.num_clients, args.niid, args.balance, args.partition):
        return

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path + '/rawdata', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path + '/rawdata', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    num_classes = len(dataset_label)
    if args.long_tail:
        dataset_image, dataset_label, num_classes = get_long_tail(dataset_image,
                                                                  dataset_label, num_classes, args.imb_factor,
                                                                  args.imb_type)

    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), args.num_clients, num_classes,
                                    args.niid, args.alpha, args.balance, args.partition)
    train_data, test_data = split_data(X, y, args.train_ratio)
    save_file(config_path, train_path, test_path, train_data, test_data, args.num_clients, num_classes,
              statistic, args.niid, args.balance, args.partition)