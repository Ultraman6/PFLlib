import torchvision
from ..dataset_utils import ConcatSet

# Allocate data to users
def _generate_dataset(dir_path, transform):
    trainset = torchvision.datasets.ImageFolder(
        root=dir_path+'/rawdata/train', transform=transform)
    validset = torchvision.datasets.ImageFolder(
        root=dir_path+'/rawdata/valid',  transform=transform)
    testset = torchvision.datasets.ImageFolder(
        root=dir_path+'/rawdata/test',  transform=transform)
    return ConcatSet([trainset, validset, testset])