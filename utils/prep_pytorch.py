import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_pretrained_model():
    batch_size = 64
    img_size = (28, 28)

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    # Replace with your actual dataset path
    train_dataset = datasets.ImageFolder(r'C:/Users/HP-PC/Desktop/Projet_Computer_Vision/data/training', transform=transform)
    test_dataset = datasets.ImageFolder(r'C:/Users/HP-PC/Desktop/Projet_Computer_Vision/data/testing', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
