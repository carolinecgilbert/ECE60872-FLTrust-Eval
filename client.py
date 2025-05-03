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
    
    def test(self, test_loader):
        """
        Test the local model on the test dataset.
        """
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x) 
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        print(f"Client {self.id} Test Accuracy: {accuracy:.4f}%")
        return accuracy

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
    
class P2PClient(Client):
    """
    Simulation of a P2P client in federated learning.
    """
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)

    def aggregate_models(self, peer_models, weights):
        new_params = [torch.zeros_like(p) for p in self.model.parameters()]
        for peer_id, peer_model in peer_models.items():
            w = weights[peer_id]
            for i, param in enumerate(peer_model.parameters()):
                new_params[i] += w * param.data
        for p, new in zip(self.model.parameters(), new_params):
            p.data.copy_(new)

    

