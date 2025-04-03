from client import Client

import copy
import torch

class Server:
    """"
    Federated Learning Server to aggregate model updates from clients.
    The default aggregation method is FedAvg.
    """
    def __init__(self, clients: list[Client], global_model, device, epochs=1):
        self.clients = clients
        self.global_model = global_model.to(device)
        self.device = device
        self.epochs = epochs

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

    def run_round_of_training(self):
        """
        Run a round of training for all clients and aggregate local updates to get global update.
        """
        client_weights = []
        for client in self.clients:
            client.model.load_state_dict(self.global_model.state_dict())
            weights = client.train(self.epochs)
            client_weights.append(weights)

        self.aggregate(client_weights)

    def test_global_model(self, test_loader):
        """
        Test the global model on the test dataset.
        """
        self.global_model.to(self.device)
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.global_model(x) 
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
    

class FLTrustServer(Server):
    """
    FLTrust server for federated learning with trust bootstrap.
    """
    def __init__(self, clients: list[Client], global_model, device, epochs=1):
        super().__init__(clients, global_model, device, epochs)

    def aggregate(self, client_weights):
        """
        Aggregation Rule: compute global model weights using FLTrust.
        """
        # TODO: Implement FLTrust aggregation logic

        

