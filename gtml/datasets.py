import torchvision.datasets as datasets
import torchvision.transforms as transforms

from gtml.constants import DATASETS_DIR


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    return (datasets.MNIST(DATASETS_DIR, train=True, transform=transform, download=True),
            datasets.MNIST(DATASETS_DIR, train=False, transform=transform, download=True))



CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return (datasets.CIFAR10(DATASETS_DIR, train=True, transform=transform_train, download=True),
            datasets.CIFAR10(DATASETS_DIR, train=False, transform=transform_test, download=True))
