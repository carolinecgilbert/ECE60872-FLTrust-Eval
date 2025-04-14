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

        # Save initial weights
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        print(f"Honest model update norm: {self.model_update_norm(initial_state, self.model.state_dict())}") 

        return self.model.state_dict()
    
    def model_update_norm(self, model_i, model_f):
        model_update_norm = 0 
        for k in model_f.keys():
            diff = model_f[k] - model_i[k]
            model_update_norm += torch.norm(diff).item() ** 2
        return model_update_norm ** 0.5


# Performs gradient ascent
# Gradient and model updates are normalized to prevent NaN values from appearing
# For a large round count, NaN values can still appear. 
# This happens due to the legitimate algorithm breaking with large values. 
class GradientAscent(Client):
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)

    def train(self, epochs=1, lr=0.01):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Train model honestly to get model update magnitude. 
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        # Calculate model update norm: 
        honest_state = self.model.state_dict()
        honest_update_norm = self.model_update_norm(initial_state, honest_state)
        
        # If trained honest model has nan, break out of here
        for k in honest_state.keys():
            if (torch.isnan(honest_state[k]).any()):
                return initial_state

        # Reset model from initial_state 
        self.model.load_state_dict(initial_state)

        max_norm = None

        # --- Gradient ascent --- #
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()

                # Clip gradient so it doesn't explode
                if max_norm is None:
                    max_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e6)
                else: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

                # Reverse gradients
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad *= -1

                optimizer.step()

        # --- Normalize model updates --- #
        mal_update_norm = self.model_update_norm(initial_state, self.model.state_dict())
        scale = honest_update_norm / mal_update_norm
        # Return honest update if gradient ascent results in nan parameter values
        if (torch.isnan(torch.tensor(scale))): 
            return honest_state
        # print(f"Scale: {honest_update_norm}/{mal_update_norm} = {scale}")
        # If honest update is larger than malicious update, don't upscale malicious update
        if (scale > 1):
            scale = 1
            return self.model.state_dict()
        else:    
            # Normalize model update using model_update_norm
            mal_state = {}
            for k in self.model.state_dict().keys():
                diff = self.model.state_dict()[k] - initial_state[k]
                diff.mul_(scale)
                mal_state[k] = initial_state[k] + diff
                if (torch.isnan(mal_state[k]).any()):
                    return honest_state
                
        print(f"GradientAscent model update norm: {self.model_update_norm(initial_state, mal_state)}") 
                    
        return mal_state

    
# Gradient ascent without the normalization
# Breaks model because model parameters blow up while training batches. This is why normalization is needed in the code above
# Not good because too obvious, sending nan parameters to central server as a malicious update is boring
class GradientAscentNoScale(Client):
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)

    def train(self, epochs=1, lr=0.01):
        self.model.train()

        # Save given model to calculate negative update later
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(self.train_data):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()

                # Negate gradients for gradient ascent
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad *= -1

                optimizer.step()

        print(f"GradientAscent model update norm: {self.model_update_norm(initial_state, self.model.state_dict())}") 
        
        return self.model.state_dict()

# Flips the model update in the opposite direction
class SignFlipping(Client):
    def __init__(self, client_id, model, dataset, indices, device):
        super().__init__(client_id, model, dataset, indices, device)

    def train(self, epochs=1, lr=0.01):
        self.model.train()

        # Save given model to calculate negative update later
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        # flip model update
        flipped_state = {}
        for k in self.model.state_dict().keys():
            delta = self.model.state_dict()[k] - initial_state[k]
            flipped_state[k] = initial_state[k] - delta 

        print(f"SignFlipping model update norm: {self.model_update_norm(initial_state, flipped_state)}") 

        return flipped_state



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