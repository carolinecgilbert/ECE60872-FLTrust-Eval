import torch
import math
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from collections import defaultdict
import random

# Hyperparameters for tracking low trust clients
LOW_TRUST_PENALTY = 0.5
LOW_TRUST_COUNT_THRESHOLD = 3
LOW_TRUST_SCORE = 0.1

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

        print(f"Honest model update norm: {Client.state_dict_update_norm(initial_state, self.model.state_dict())}") 

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
      
    
    # Takes a initial model state_dict and the model state_dict after an update. Calculates the magnitude of the update. 
    @staticmethod
    def state_dict_update_norm(model_i, model_f):
        norm = 0 
        for k in model_f.keys():
            diff = model_f[k] - model_i[k]
            norm += torch.norm(diff).item() ** 2
        return norm ** 0.5
        
    @staticmethod
    def state_dict_norm(model):
        norm = 0
        for k in model.keys():
            norm += torch.norm(model[k]).item() ** 2
        return norm ** 0.5
    
    @staticmethod
    def state_dict_normalize(model_i, model_f, norm):
        if (model_i is not None):
            update = {k: model_f[k] - model_i[k] for k in model_f.keys()}

            update_norm = Client.state_dict_update_norm(model_i, model_f)
            scale = norm / update_norm
            
            model_n = {}
            for k in update.keys():
                update[k].mul_(scale)
                model_n[k] = model_i[k] + update[k]
        else:
            update_norm = Client.state_dict_norm(model_f)
            scale = norm / update_norm
            
            model_n = {k: v.clone() for k, v in model_f.items()}
            for k in model_n.keys():
                model_n[k].mul_(scale)

        return (model_n, scale)
        
    
    @staticmethod
    def state_dict_cosine_sim(model_u, model_x):
        u_vector = torch.cat([v.flatten() for v in model_u.values()])
        x_vector = torch.cat([v.flatten() for v in model_x.values()])

        # Compute cosine similarity
        return F.cosine_similarity(u_vector.unsqueeze(0), x_vector.unsqueeze(0)).item()
    
    @staticmethod
    def state_dict_dot_product(model_u, model_x):
        # Flatten and concatenate all tensors
        u_vector = torch.cat([v.flatten() for v in model_u.values()])
        x_vector = torch.cat([v.flatten() for v in model_x.values()])

        # Compute dot product
        dot_product = torch.dot(u_vector, x_vector).item()

        return dot_product
    
    # Returns model with updates at the given angle from model_u. Uses model_x to generate a orthogonal model
    @staticmethod
    def state_dict_rotate(init_model, model_u, model_x, angle):
        # Compute model updates for module u and module x
        u_update = {k: model_u[k] - init_model[k] for k in model_u.keys()}
        x_update = {k: model_x[k] - init_model[k] for k in model_x.keys()}
        u_update_norm = Client.state_dict_norm(u_update)

        # u.x / u.u
        scale = Client.state_dict_dot_product(u_update, x_update) / Client.state_dict_dot_product(u_update, u_update)

        # Scale u update to get the projection of x onto u. 
        model_p = {k: v.clone() for k, v in u_update.items()}
        for k in model_p.keys():
            model_p[k].mul_(scale)
        (model_pn, _) = Client.state_dict_normalize(None, model_p, u_update_norm)

        # Subtract projection from x to get the orthogonal projection
        model_o = {}
        for k in model_p.keys():
            model_o[k] = x_update[k] - model_p[k]
        (model_on, _) = Client.state_dict_normalize(None, model_o, u_update_norm)


        # Combine parallel and orthogonal vectors to reach given angle from model_u
        rotated_update = {}
        for k in u_update.keys():
            rotated_update[k] = init_model[k] + model_pn[k].mul_(math.cos(angle)) + model_on[k].mul_(math.sin(angle))

        return rotated_update

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
    """
    Simulation of a P2P Client with trust scores in FL.
    """
    def __init__(self, client_id, model, dataset, indices, device, apply_penalties=False):
        super().__init__(client_id, model, dataset, indices, device)
        self.prev_params = None  
        self.trust_history = defaultdict(list)
        self.apply_penalties = apply_penalties

    def aggregate_models(self, peer_clients, weights):
        """
        Aggregate models from peers using cosine similarity-based trust scores.
        """
        # Compute local model update
        local_init = {k: p.clone().detach() for k, p in zip(self.model.state_dict().keys(), self.prev_params)}
        local_final = self.model.state_dict()
        local_update_norm = Client.state_dict_update_norm(local_init, local_final)

        trust_scores = {}
        normalized_peer_models = {}

        for pid, peer in peer_clients.items():
            # if pid == self.id:
            #     continue

            # Compute peer model update 
            peer_init = {k: p.clone().detach() for k, p in zip(peer.model.state_dict().keys(), peer.prev_params)}
            peer_final = peer.model.state_dict()

            # Compute trust score via cosine similarity between local and peer update (FLTrust)
            cosine_sim = Client.state_dict_cosine_sim(local_final, peer_final)
            trust_score = max(0.0, cosine_sim)  
            trust_scores[pid] = trust_score

            # Apply penalty for repeated low trust (Novel FLTrust)
            if self.apply_penalties:
                low_count = sum(1 for s in self.trust_history[pid] if s < LOW_TRUST_SCORE)
                if low_count >= LOW_TRUST_COUNT_THRESHOLD:
                    print(f"Client {self.id} penalizing peer {pid} for low trust.")
                    trust_score *= LOW_TRUST_PENALTY
                    trust_scores[pid] = trust_score

            # Update trust history
            self.trust_history[pid].append(trust_score)

            # Normalize peer update to match local update norm (FLTrust)
            normed_peer_model, _ = Client.state_dict_normalize(peer_init, peer_final, local_update_norm)
            normalized_peer_models[pid] = normed_peer_model

        # Weighted aggregation using trust 
        total_trust = sum(trust_scores.values())
        if total_trust < 1e-10:
            print(f"Client {self.id} skipping aggregation (total trust weight too small).")
            return

        update_sum = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        for pid, peer_model in normalized_peer_models.items():
            ts = trust_scores[pid]
            for k in update_sum:
                update_sum[k] += ts * (peer_model[k] - self.model.state_dict()[k])

        # Aggregate the updates
        for k in self.model.state_dict().keys():
            self.model.state_dict()[k].add_(update_sum[k] / total_trust)

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
    

class Backdoor(Client):

    def __init__(self, client_id, model, dataset, indices, device, trigger_pattern = [[-.42, -.42, -.42, -.42, -.42], 
                                                                                      [-.42, 2.80, 2.80, 2.80, -.42], 
                                                                                      [-.42, 2.80, -.42, 2.80, -.42], 
                                                                                      [-.42, 2.80, -.42, 2.80, -.42], 
                                                                                      [-.42, 2.80, 2.80, 2.80, -.42], 
                                                                                      [-.42, -.42, -.42, -.42, -.42]], trigger_output = 0):
        super().__init__(client_id, model, dataset, indices, device)

        self.trigger_pattern = torch.tensor(trigger_pattern)    # Pattern that triggers the backdoor
        self.trigger_output = torch.tensor(trigger_output)      # Label model should flip to when pattern exists in the input image

    @staticmethod
    def add_trigger(tensor, trigger, random_offset = False):

        # print(f"Tensor shape: {tensor.shape}, trigger shape: {trigger.shape}")

        trigger_h, trigger_w = trigger.shape
        tensor_h, tensor_w = tensor.shape

        if (random_offset):
            h_off = random.randint(0, tensor_h - trigger_h)
            w_off = random.randint(0, tensor_w - trigger_w)
        else:
            h_off = 0
            w_off = 0

        tensor[h_off:h_off+trigger_h, w_off:w_off+trigger_w] = trigger

        return tensor

    # Consider training once per iteration (Either with backdoor or without. Not both)
    def train(self, epochs=1, lr=0.01):
        self.model.train()

        # Save initial weights
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        self.trigger_output = self.trigger_output.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:

                # Create backdoor input images
                x_bt = x.detach().clone()
                for image in x_bt:
                    image[0] = Backdoor.add_trigger(image[0], self.trigger_pattern)

                x, x_bt, y = x.to(self.device), x_bt.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # Train with backdoor
                output = self.model(x_bt)
                loss = F.cross_entropy(output, torch.full_like(y, self.trigger_output))
                loss.backward()
                optimizer.step()

                # Train normally
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        print(f"Backdoor update norm: {Client.state_dict_update_norm(initial_state, self.model.state_dict())}") 

        return self.model.state_dict()
      


class LabelFlipping(Client):
    #                                                                         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    def __init__(self, client_id, model, dataset, indices, device, labelmap = [8, 1, 5, 8, 1, 5, 8, 1, 8, 8]):
        super().__init__(client_id, model, dataset, indices, device)
        
        self.labelmap = torch.tensor(labelmap)

    def train(self, epochs=1, lr=0.01):
        self.model.train()

        # Save initial weights
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.labelmap = self.labelmap.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                y = self.labelmap[y]
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()

        print(f"Label flipping model update norm: {Client.state_dict_update_norm(initial_state, self.model.state_dict())}") 

        return self.model.state_dict()
    
    def state_dict_update_norm(self, model_i, model_f):
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
    def __init__(self, client_id, model, dataset, indices, device, mal_angle = 0):
        super().__init__(client_id, model, dataset, indices, device)
        self.mal_angle = mal_angle * math.pi / 180

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
        honest_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        honest_update_norm = Client.state_dict_update_norm(initial_state, honest_state)
        
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
        mal_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        if (self.mal_angle == 0):
            #normalize model update to the same magnitude as honest update
            (norm_state, scale) = Client.state_dict_normalize(initial_state, mal_state, honest_update_norm)
            print(f"Grad ascent norm: {Client.state_dict_update_norm(initial_state, mal_state)}, scale: {scale}")

            # Return honest update if mal update has nan
            if (torch.isnan(torch.tensor(scale))):
                return honest_state
            # If honest update is larger than malicious update, don't upscale malicious update
            if (scale > 1):
                return self.model.state_dict()
            else:    
                for k in norm_state.keys():
                    if (torch.isnan(norm_state[k]).any()):
                        return honest_state
                return norm_state
        # If set cosine angle, normalize magniude and modify angle
        else: 
            return Client.state_dict_rotate(initial_state, honest_state, mal_state, self.mal_angle)

    
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

        print(f"GradientAscent model update norm: {Client.state_dict_update_norm(initial_state, self.model.state_dict())}") 
        
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

        print(f"SignFlipping model update norm: {Client.state_dict_update_norm(initial_state, flipped_state)}") 

        return flipped_state

