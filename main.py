import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision import datasets

def train():
    return


def test():
    return


def main():
    #########################
    # load CIFAR-10 dataset #
    #########################
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    train_data = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)



if __name__ == '__main__':
    main()