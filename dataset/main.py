import argparse
import os
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from utils.partition import *

class OnlyOneAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, default=None, **kwargs):
        # 初始化时接收默认值
        super().__init__(option_strings, dest, nargs, **kwargs)
        self.default = default  # 设置默认值
        self.values = []

    def __call__(self, parser, namespace, values, option_string=None):
        # 每次调用时将所有的值存储到values列表
        if not self.values:
            self.values = values  # 只第一次接收到这些值

        # 如果列表为空，使用默认值
        if not self.values:
            setattr(namespace, self.dest, self.default)
        else:
            # 获取并移除列表中的第一个值
            current_value = self.values.pop(0)  # 返回并移除首位元素
            setattr(namespace, self.dest, current_value)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, nargs='+', type=int)
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--root', default='../../data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('-pt', '-partition', default='iid', nargs='+', type=str, help='Partition type')
    parser.add_argument('-nc', '-num_clients', default=20, type=int,  action=OnlyOneAction)
    parser.add_argument('-np', '-num_per', default=None, nargs='+', type=int, action=OnlyOneAction)
    parser.add_argument('-cp', '-cls_per', default=-1, nargs='+', type=int,  action=OnlyOneAction)
    parser.add_argument('-nm', '-num_map', default=None, nargs='+', type=int, action=OnlyOneAction)
    parser.add_argument('-cm', '-cls_map', default=None, nargs='+', type=int, action=OnlyOneAction)
    parser.add_argument('-im', '-imbalance', default=0., nargs='+', type=float, action=OnlyOneAction)
    parser.add_argument('-a',  '-alpha', default=1.0, nargs='+', type=float,  action=OnlyOneAction)
    parser.add_argument('-e',  '-error_bar', default=1e-6, nargs='+', type=float, action=OnlyOneAction)
    parser.add_argument('-m',  '-minvol', default=32, nargs='+', type=int, action=OnlyOneAction)
    parser.add_argument('-d',  '-disturb', default=1.0, nargs='+', type=float, action=OnlyOneAction)
    parser.add_argument('-it', '-imb_type', default='exp', nargs='+', type=str, action=OnlyOneAction)
    parser.add_argument('-ir', '-imb_factor', default=0.01, nargs='+', type=float, action=OnlyOneAction)
    parser.add_argument('-co', '-cls_order', default='random', nargs='+', type=str, action=OnlyOneAction)
    args = parser.parse_args()
    print(args.new)
    return args

def get_partitioner(pt, args):
    kwargs = vars(BasicPartitioner(num_clients=args.num_clients, num_per=args.np, cls_per=args.cp))
    partitioners = {
        'iid': IIDPartitioner(imbalance=args.im, num_clients=args.nc, **kwargs),
        'dir': DirichletPartitioner(alpha=args.a, error_bar=args.e, imbalance=args.im, minvol=args.m, **kwargs),
        'exdir': ExDirichletPartitioner(alpha=args.a, minvol=args.m, **kwargs),
        'pat': PathologyPartitioner(imbalance=args.im, disturb=args.d, **kwargs),
        'custom': CustomPartitioner(cls_map=args.cm, num_map=args.nm, minvol=args.m, **kwargs)
    }
    if pt in partitioners:
        return partitioners[pt]
    else:
        raise ValueError(f"Unsupported partitioner: {pt}")

# Allocate data to users
def generate_dataset(args):
    dir_path = str(os.path.join(args.root, args.dataset))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if args.dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        from utils.DATA.cifar10 import _generate_dataset

    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        from utils.DATA.cifar100 import _generate_dataset

    elif args.dataset == 'fminist':
        transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.2860366729433025), (0.35288708155778725))
            ])
        from utils.DATA.fmnist import _generate_dataset

    elif args.dataset == 'cinic10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        from utils.DATA.cinic10 import _generate_dataset

    elif args.dataset == 'domainnet':
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.75, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transforms_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        transform = (transforms_train, transforms_test)
        from utils.DATA.domainnet import _generate_dataset

    elif args.dataset == 'tinyimagenet':
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std)])
        from utils.DATA.tinyimagenet import _generate_dataset

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    dataset = _generate_dataset(dir_path, transform)
    pt = args.pt
    if pt == 'long_tail':
        partitioner = get_partitioner(args.pt, args)
        partitioner = LongTailPartitioner(partitioner, imb_type=args.it, imb_factor=args.ir,  cls_order=args.cls_order)
    elif pt == 'hierarch':
        partitioner1 = get_partitioner(args.pt, args)
        partitioner2 = get_partitioner(args.pt, args)
        partitioner = HierarchPartitioner(partitioner1, partitioner2)
    elif pt == 'label_domain':
        partitioner1 = get_partitioner(args.pt, args)
        partitioner2 = get_partitioner(args.pt, args)
        partitioner = LabelDomainPartitioner(partitioner1, partitioner2)
    else:
        partitioner = get_partitioner(pt, args)
    info = partitioner.get_info()
    path = os.path.join(dir_path, info)
    if args.new or not os.path.exists(path):
        data, distribution = partitioner(dataset)
        train_data, test_data = split_data(data, args.train_ratio)
        save_file(path, train_data, test_data, distribution)

    return info


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    info = generate_dataset(args)
    print("Dataset generated successfully! in {}".format(info))