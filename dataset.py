import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

class MNISTDataset:
    """
    MNIST Dataset class to handle loading the dataset for training
    and testing.
    """
    def __init__(self, num_clients=10):
        self.num_clients = num_clients
        self.train_dataset = self.get_train_dataset()
        self.test_dataset = self.get_test_dataset()
        
    def get_train_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    def get_test_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    def partition_train_dataset(self, num_clients):
        data_size = len(self.train_dataset)
        indices = list(range(data_size))
        random.shuffle(indices)
        return [indices[i::num_clients] for i in range(num_clients)]
    
    def visualize_samples(self, num_images=3):
        loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=num_images, shuffle=True)
        images, labels = next(iter(loader))
        plt.figure(figsize=(12, 2))
        for i in range(num_images):
            plt.subplot(1, num_images, i+1)
            plt.imshow(images[i].squeeze(), cmap="gray")
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("mnist_samples.png")