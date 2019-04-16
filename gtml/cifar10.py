import torchvision.datasets as datasets
import torchvision.transforms as transforms

from gtml.constants import DATASETS_DIR


PIXEL_MEAN = (0.4914, 0.4822, 0.4465)
PIXEL_STD = (0.2470, 0.2435, 0.2616)

def load_train():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])
    return datasets.CIFAR10(DATASETS_DIR, train=True, transform=transform,
                            download=True)

def load_test():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])
    return = datasets.CIFAR10(DATASETS_DIR, train=False, transform=transform,
                              download=True)
