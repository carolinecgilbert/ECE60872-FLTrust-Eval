import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

TRUST_THRESHOLD = 0.1

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
                loss.backward()
                optimizer.step()

        # Dummy attack: arbitrarily scale model parameters (malicious update)
        with torch.no_grad():
            for p in self.model.parameters():
                p.data *= 5.0  # exaggerate the model weights
            return self.model.state_dict()
    
class P2PClient(Client):
    """
    Simulation of a P2P client in federated learning.
    """
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)
        self.prev_params = None

    def store_local_update(self):
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]

    def get_local_update(self):
        return torch.cat([(p.data - prev).view(-1) for p, prev in zip(self.model.parameters(), self.prev_params)])
    
    def extract_update(self, peer):
        """
        Compute the update vector of a peer using the peer's own stored parameters.
        """
        return [(param.data - old_param).detach().clone()
                for param, old_param in zip(peer.model.parameters(), peer.prev_params)]


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
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)
        self.prev_params = None  
        self.trust_history = {}

    def aggregate_models(self, peer_clients, weights):
        g_self = self.get_local_update()
        g_self_norm = torch.norm(g_self)

        trust_scores = {}
        translated_updates = {}

        for pid, peer in peer_clients.items():
            if pid == self.id:
                continue  # skip self in trust comparison

            peer_prev_params = peer.prev_params
            peer_curr_params = list(peer.model.parameters())
            delta_peer = [curr.data - prev for curr, prev in zip(peer_curr_params, peer_prev_params)]

            # Translate peer update to local context
            translated = [p.data + delta for p, delta in zip(self.model.parameters(), delta_peer)]
            translated_update = [t - p.data for t, p in zip(translated, self.
            model.parameters())]

            g_peer = torch.cat([u.view(-1) for u in translated_update])

            g_peer_norm = torch.norm(g_peer)
            if g_peer_norm > 0 and g_self_norm > 0:
                cosine_sim = torch.dot(g_peer, g_self) / (g_peer_norm * g_self_norm)
                trust_scores[pid] = F.relu(cosine_sim).item()
            else:
                trust_scores[pid] = 0.0

            translated_updates[pid] = translated_update

        # Update trust history
        for pid, score in trust_scores.items():
            if pid not in self.trust_history:
                self.trust_history[pid] = []
            self.trust_history[pid].append(score)

        weighted_sum = [torch.zeros_like(p) for p in self.model.parameters()]
        total_weight = sum(trust_scores[pid] * weights[pid] for pid in trust_scores)

        if total_weight < 1e-10:
            print(f"Client {self.id} skipping aggregation (total trust weight too small).")
            return

        for pid, update in translated_updates.items():
            if pid not in trust_scores or trust_scores[pid] < TRUST_THRESHOLD:
                continue
            weight = trust_scores[pid] * weights[pid]
            for i in range(len(weighted_sum)):
                weighted_sum[i] += weight * update[i]

        for p, delta in zip(self.model.parameters(), weighted_sum):
            p.data += delta / total_weight



class MaliciousP2PFLTrustClient(P2PFLTrustClient):
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
                loss.backward()
                optimizer.step()

        # Dummy attack: arbitrarily scale model parameters (malicious update)
        with torch.no_grad():
            for p in self.model.parameters():
                p.data *= 5.0  # exaggerate the model weights
            return self.model.state_dict()
    

