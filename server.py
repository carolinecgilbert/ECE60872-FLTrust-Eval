import copy
import torch

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

# TODO: Implement FLTrust server and other novel aggregation methods