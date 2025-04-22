
import torchvision
from ..dataset_utils import ConcatSet

# Allocate data to users
def _generate_dataset(dir_path, transform):
    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+'/rawdata', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path+'/rawdata', train=False, download=True, transform=transform)
    return ConcatSet([trainset, testset])