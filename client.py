import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

class Client:
    """
    Simulation of a client in a federated learning.
    Each client has its own model, dataset, train function.
    """
    def __init__(self, client_id, model, dataset, indices, device):
        self.id = client_id
        self.device = device
        self.model = model.to(self.device)
        self.train_data = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True)

    def train(self, epochs=1, lr=0.01):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()

class MaliciousClient(Client):
    """"
    Simulation of a malicious client in a federated learning.
    This client injects attacks during model training.
    """
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)

    def train(self, epochs=1, lr=0.01):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                # Basic attack: reverse the gradient for malicious behavior
                # TODO: Implement an attack class for more complex attacks
                loss = -loss 
                loss.backward()
                optimizer.step()
        return self.model.state_dict()