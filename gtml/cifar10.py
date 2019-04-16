import torchvision.datasets as datasets
import torchvision.transforms as transforms

from gtml.constants import DATASETS_DIR


def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(DATASETS_DIR, train=True, transform=transform_train, download=True)
    test_set = datasets.CIFAR10(DATASETS_DIR, train=False, transform=transform_test, download=True)
    return train_set, test_set