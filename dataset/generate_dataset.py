import argparse
import numpy as np
import random
import torchvision.transforms as transforms


# Allocate data to users
def generate_dataset(args):
    if args.dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        from dataset.DATA.generate_Cifar10 import _generate_dataset

    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        from dataset.DATA.generate_Cifar100 import _generate_dataset

    elif args.dataset == 'fminist':
        transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.2860366729433025), (0.35288708155778725))
            ])
        from generate_fmnist import _generate_dataset

    elif args.dataset == 'cinic10':
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean,std=cinic_std)])
        from generate_Cinic import _generate_dataset

    elif args.dataset == 'domainNet':
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean,std=cinic_std)])
        from dataset.DATA.generate_DomainNet import _generate_dataset
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    if args.pt == 'iid':
    elif args.pt == 'dir':
    elif args.pt == 'exdir':
    elif args.pt == 'pat':
    elif args.pt == 'div':

    _generate_dataset(args, transform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-nd', '-niid', action='store_true', type=bool, default=False)
    parser.add_argument('-bl', '-balance', action='store_true', type=bool, default=True)
    parser.add_argument('-nc', "--num_clients", type=int, default=10)
    parser.add_argument('-lt', '--long_tail', action='store_true', type=bool, default=False)
    parser.add_argument('-if', '--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('-it', '--imb_type', type=str, default='exp')
    parser.add_argument('-pt', '--partition', type=str, default='dir')
    parser.add_argument('-dr', '-data_root', type=str, default='../../data')
    parser.add_argument('-bs', '-batch_size', type=int, default=32)
    parser.add_argument('-tr', '-train_ratio', type=float, default=0.75)
    parser.add_argument('-a', '-alpha', nargs='+', type=float, default=0.)
    parser.add_argument('-cls', 'class_per_client', nargs='+', type=int, default=2)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    _generate_dataset(args)