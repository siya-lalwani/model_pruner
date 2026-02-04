from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(dataset_name, batch_size):

    # Common transform for both datasets
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 🔴 FIX: 1 channel → 3 channels
            transforms.ToTensor()
        ])

        trainset = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        )
        testset = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        trainset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform
        )
        testset = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

    else:
        raise ValueError("Unknown dataset")

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
