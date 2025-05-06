import copy
import torch

from client import *
import matplotlib.pyplot as plt

def model_distance(model_a, model_b):
    """Compute L2 distance between two model state_dicts."""
    distance = 0.0
    for k in model_a.keys():
        distance += torch.norm(model_a[k] - model_b[k]) ** 2
    return distance.item()

def consensus_error(clients):
    """
    Compute the consensus distance between the models of all clients.
    """
    with torch.no_grad():
        # Compute the average model parameters
        avg_params = [torch.zeros_like(p) for p in clients[0].model.parameters()]
        for client in clients:
            for i, p in enumerate(client.model.parameters()):
                avg_params[i] += p.data
        for i in range(len(avg_params)):
            avg_params[i] /= len(clients)

        # Compute the consensus error
        consensus_error = 0.0
        for client in clients:
            error = 0.0
            for i, p in enumerate(client.model.parameters()):
                error += torch.norm(p.data - avg_params[i])**2
            consensus_error += error.item()
        consensus_error /= len(clients)
        return consensus_error 

class Server:
    """"
    Federated Learning Server to aggregate model updates from clients.
    The default aggregation method is FedAvg.
    """
    def __init__(self, global_model, device):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_weights):
        """
        Aggregation Rule: compute global model weights using FedAvg.
        """
        new_state = copy.deepcopy(client_weights[0])
        for key in new_state:
            for i in range(1, len(client_weights)):
                new_state[key] += client_weights[i][key]
            new_state[key] = torch.div(new_state[key], len(client_weights))
        self.global_model.load_state_dict(new_state)

    def test_global_model(self, test_loader):
        """
        Test the global model on the test dataset.
        """
        self.global_model.to(self.device)
        self.global_model.eval()
        correct = 0
        total = 0

        # Backdoor trigger pattern
        trigger_pattern = torch.tensor([[-.42, -.42, -.42, -.42, -.42], 
                                        [-.42, 2.80, 2.80, 2.80, -.42], 
                                        [-.42, 2.80, -.42, 2.80, -.42], 
                                        [-.42, 2.80, -.42, 2.80, -.42], 
                                        [-.42, 2.80, 2.80, 2.80, -.42], 
                                        [-.42, -.42, -.42, -.42, -.42]])

        with torch.no_grad():
            for x, y in test_loader:

                # Add backdoor pattern
                # for image in x:
                #     image[0] = Backdoor.add_trigger(image[0], trigger_pattern)

                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x) 
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

class P2PServer():
    """
    P2P Federated Learning Server to aggregate model updates from clients
    using FL trust.
    """
    def __init__(self, clients, model, device):
        self.clients = clients
        self.model = model.to(device)
        self.device = device

    def test(self, test_loader):
        """
        Test the global model on the test dataset.
        """
        self.model.to(self.device)
        self.model.eval()
        correct = 0
        total = 0

        # Backdoor trigger pattern
        trigger_pattern = torch.tensor([[-.42, -.42, -.42, -.42, -.42], 
                                        [-.42, 2.80, 2.80, 2.80, -.42], 
                                        [-.42, 2.80, -.42, 2.80, -.42], 
                                        [-.42, 2.80, -.42, 2.80, -.42], 
                                        [-.42, 2.80, 2.80, 2.80, -.42], 
                                        [-.42, -.42, -.42, -.42, -.42]])

        with torch.no_grad():
            for x, y in test_loader:

                # Add backdoor pattern
                # for image in x:
                #     image[0] = Backdoor.add_trigger(image[0], trigger_pattern)

                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x) 
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def compute_final_trust_scores(self):
        """
        Compute overall trust scores for each client
        based on how peers trusted them over all rounds.
        """
        # Aggregate trust scores from all clients
        peer_ids = [client.id for client in self.clients]
        trust_agg = {pid: [] for pid in peer_ids}
        for client in self.clients:
            for pid, scores in client.trust_history.items():
                trust_agg[pid].extend(scores)

        # Compute final trust scores
        final_scores = {}
        for pid, scores in trust_agg.items():
            if scores:
                final_scores[pid] = sum(scores) / len(scores)
            else:
                final_scores[pid] = 0.0
        return final_scores
    
    def select_top_trusted_clients(self, final_scores, top_percent=0.6):
        """
        Select top_percent % of clients based on final aggregated trust scores.
        """
        num_selected = max(1, int(len(self.clients) * top_percent))
        sorted_clients = sorted(self.clients, key=lambda c: final_scores.get(c.id, 0), reverse=True)
        return sorted_clients[:num_selected]
    
    def aggregate_trusted_models(self):
        """
        Given a list of trusted clients and their final models,
        check consensus and average the models weighted by trust.
        """
        if len(self.clients) == 0:
            raise ValueError("No trusted clients to aggregate from.")
        
        # Compute final trust scores
        final_scores = self.compute_final_trust_scores()
        trusted_clients = self.select_top_trusted_clients(final_scores)

        if len(trusted_clients) == 0:
            raise ValueError("No trusted clients to aggregate from.")
        else:
            print(f"Selected {len(trusted_clients)} trusted clients for aggregation: {[c.id for c in trusted_clients]}")

        # Validate model similarity using consensus distance
        # TODO: Future work can track clients with high consensus distances with trusted scores to see if it is possible to find an anomaly
        models = [client.model.state_dict() for client in trusted_clients]
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                dist = model_distance(models[i], models[j])
                if dist > 1e-1:
                    print(f"WARNING: Model {i} and {j} have high consensus distance: {dist:.4f}")

        # Normalize trust scores for weighting
        total = sum(final_scores[c.id] for c in trusted_clients)
        normalized_weights = {c.id: final_scores[c.id] / total for c in trusted_clients}

        # Aggregate models using weighted average
        avg_model = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        for client in trusted_clients:
            weight = normalized_weights[client.id]
            model_state = client.model.state_dict()
            for k in avg_model:
                avg_model[k] += weight * model_state[k]

        # Load the aggregated model into the global model
        self.model.load_state_dict(avg_model)

