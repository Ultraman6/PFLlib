import torchvision
from ..dataset_utils import ConcatSet

# Allocate data to users
def _generate_dataset(dir_path, transform):
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+'/rawdata', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+'/rawdata', train=False, download=True, transform=transform)
    return ConcatSet([trainset, testset])