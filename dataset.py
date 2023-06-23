import torch
from torchvision import transforms, datasets
from utils import is_cuda_available

def get_train_test_loaders():

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    data_loader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if is_cuda_available() else dict(shuffle=True, batch_size=128)

    train_loader = torch.utils.data.DataLoader(train_dataset, **data_loader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **data_loader_args)

    return train_loader, test_loader