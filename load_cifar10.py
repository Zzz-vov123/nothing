import torch
import torchvision
import d2l
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

def load_cifar10(is_train):
    if is_train==True:
        augs=train_augs
    else:
        augs=test_augs
    dataset = torchvision.datasets.CIFAR10(root="data/", train=is_train,
                                           transform=augs, download=False)
    return dataset
