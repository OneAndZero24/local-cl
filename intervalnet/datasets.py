import os
import warnings

import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST, FashionMNIST
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST


def split_mnist(n_experiences: int):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return SplitMNIST(
        n_experiences=n_experiences,
        train_transform=transform,
        eval_transform=transform
    )

def split_cifar10(n_experiences: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    return SplitCIFAR10(
        n_experiences=n_experiences,
        train_transform=transform,
        eval_transform=transform
    )

def mnist():
    """MNIST dataset."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mnist_train = MNIST(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        mnist_test = MNIST(root=os.getenv('DATA_DIR', ''), train=False, download=True)
    mnist_train_transform = transforms.Compose([
        # transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # for 28x28
        # transforms.Normalize(mean=(0.1000,), std=(0.2752,)),  # for 32x32
    ])
    mnist_eval_transform = transforms.Compose([        # transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
    ])
    return mnist_train, mnist_test, mnist_train_transform, mnist_eval_transform


def cifar100():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cifar_train = CIFAR100(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        cifar_test = CIFAR100(root=os.getenv('DATA_DIR', ''), train=False, download=True)
    cifar_train_transform = transforms.Compose([
        # transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cifar_eval_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return cifar_train, cifar_test, cifar_train_transform, cifar_eval_transform


def cifar10():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cifar_train = CIFAR10(root=os.getenv('DATA_DIR', ''), train=True, download=True)
        cifar_test = CIFAR10(root=os.getenv('DATA_DIR', ''), train=False, download=True)
    cifar_train_transform = transforms.Compose([
        # transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    cifar_eval_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    return cifar_train, cifar_test, cifar_train_transform, cifar_eval_transform


def fashion_mnist():
    """Fashion-MNIST dataset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train = FashionMNIST(root=os.getenv("DATA_DIR", ""), train=True, download=True)
        test = FashionMNIST(root=os.getenv("DATA_DIR", ""), train=False, download=True)
    transforms_ = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return train, test, transforms_, transforms_
