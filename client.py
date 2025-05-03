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
        self.prev_params = None

    def store_local_update(self):
        """
        Store the local update for the client.
        """
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]

    def get_local_update(self):
        """
        Get the local update for the client.
        """
        return torch.cat([(p.data - prev).view(-1) for p, prev in zip(self.model.parameters(), self.prev_params)])

    def aggregate_models(self, peer_models, weights):
        """
        Aggregate models from peers using a weighted sum.
        """
        new_params = [torch.zeros_like(p) for p in self.model.parameters()]
        for peer_id, peer_model in peer_models.items():
            w = weights[peer_id]
            for i, param in enumerate(peer_model.parameters()):
                new_params[i] += w * param.data
        for p, new in zip(self.model.parameters(), new_params):
            p.data.copy_(new)


class P2PFLTrustClient(P2PClient):
    """
    Simulation of a P2PFLTrust client in federated learning.
    """
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)
        self.prev_params = None

    def aggregate_models(self, peer_models, weights):
        """
        Aggregate models from peers using a weighted sum and trust scores as in FLTrust.
        """
        # Compute trust scores and normalize updates
        local_update = self.get_local_update()
        peer_updates = {}
        for pid, model in peer_models.items():
            peer_updates[pid] = self.extract_update(model)
        trust_scores = self.compute_trust_scores(peer_updates, local_update)
        normalized_updates = self.normalize_updates(peer_updates, local_update)

        # Normalize the updates
        weighted_sum = [torch.zeros_like(p) for p in self.model.parameters()]
        norm_factor = sum(trust_scores[pid] * weights[pid] for pid in peer_models)
        norm_factor = max(norm_factor, 1e-10)

        for pid, norm_update in normalized_updates.items():
            for i, param in enumerate(weighted_sum):
                param += (trust_scores[pid] * weights[pid]) * norm_update[i]

        # Update the model parameters
        for p, new in zip(self.model.parameters(), weighted_sum):
            p.data -= new / norm_factor

    def extract_update(self, model):
        return [param.data - base.data for param, base in zip(model.parameters(), self.model.parameters())]

    def compute_trust_scores(self, peer_updates, local_update):
        """
        Compute trust scores based on cosine similarity between local update
        and peer updates.
        """
        trust_scores = {}
        g0_norm = torch.norm(local_update)
        for cid, update in peer_updates.items():
            update_flat = torch.cat([u.view(-1) for u in update])  # flatten list of tensors
            gi_norm = torch.norm(update_flat)
            cosine_sim = torch.dot(update_flat, local_update) / (gi_norm * g0_norm + 1e-10)
            trust_scores[cid] = F.relu(cosine_sim).item()
        return trust_scores

    def normalize_updates(self, peer_updates, local_update):
        """
        Normalize peer updates based on the local update.
        """
        g0_norm = torch.norm(local_update)
        normalized = {}
        for cid, update in peer_updates.items():
            update_flat = torch.cat([u.view(-1) for u in update])
            gi_norm = torch.norm(update_flat)
            if gi_norm > 0:
                scaled = [(g0_norm / gi_norm) * u for u in update]
                normalized[cid] = scaled
            else:
                normalized[cid] = [torch.zeros_like(u) for u in update]
        return normalized

    

