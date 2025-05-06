from client import *
from server import Server
from dataset import MNISTDataset
from model import CNN

import torch
import sys
import time
import copy

# Build the mixing matrix for P2PFL with stochastic rows and columns
def build_mixing_matrix(neighbors_dict):
    # neighbors_dict: dict of client_id -> list of neighbor client_ids
    n = len(neighbors_dict)
    W = torch.zeros((n, n))
    for i in range(n):
        for j in neighbors_dict[i]:
            W[i, j] = 1.0 / max(len(neighbors_dict[i]) + 1, len(neighbors_dict[j]) + 1)
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W

# Check consensus by computing average pairwise L2 distance
def model_distance(m1, m2):
    return sum(torch.norm(p1.data - p2.data).item() for p1, p2 in zip(m1.parameters(), m2.parameters()))

def compute_consensus_distance(clients):
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

if __name__ == "__main__":
    # Default client set up
    num_clients = 5
    num_malicious = 0
    rounds = 5
    penalize_low_trust = False

    # Check for command line arguments to change client setup
    if len(sys.argv) > 1:
        num_clients = max(int(sys.argv[1]), 1)
    if len(sys.argv) > 2:
        num_malicious = min(int(sys.argv[2]), num_clients)
    if len(sys.argv) > 3:
        rounds = max(int(sys.argv[3]), 1)
    if len(sys.argv) > 4:
        if sys.argv[4].lower() == 'true':
            penalize_low_trust = True
    print(f"Number of clients: {num_clients}")
    print(f"Number of malicious clients: {num_malicious}")
    print(f"Number of rounds: {rounds}")
    print(f"Penalize low trust: {penalize_low_trust}")

    # Initialize datasets
    dataset = MNISTDataset()
    train_dataset = dataset.get_train_dataset()
    client_indices = dataset.partition_train_dataset(num_clients)

    # Set the device for compatibility with MacOS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize clients
    clients = []
    model = CNN()
    for i, idxs in enumerate(client_indices):
        if i < num_malicious:
            clients.append(MaliciousP2PFLTrustClient(i, copy.deepcopy(model), train_dataset, idxs, device))
        else:
            clients.append(P2PFLTrustClient(i, copy.deepcopy(model), train_dataset, idxs, device))

    # Initialize the mixing matrix for P2PFL
    neighbors_dict = {i: [j for j in range(num_clients) if j != i] for i in range(num_clients)}
    mixing_matrix = build_mixing_matrix(neighbors_dict)
    print("Mixing matrix:\n", mixing_matrix)

    # Simulate federated learning rounds
    print("Starting P2PFL simulation...")
    start_time = time.time()
    for r in range(rounds):
        # Store local updates
        for client in clients:
            client.store_local_update()

        # Local training
        for client in clients:
            client.train()


        # Aggregation using mixing matrix
        for client in clients:
            neighbors = neighbors_dict[client.id]
            peer_clients = {}
            weights = {}
            for nid in neighbors + [client.id]:
                peer_clients[nid] = clients[nid]
                weights[nid] = mixing_matrix[client.id, nid].item()
            client.aggregate_models(peer_clients, weights)


        print(f"Round {r+1} complete")

    end_time = time.time()
    print(f"P2PFL simulation completed in {end_time - start_time:.4f} seconds")

    # Consensus distance computation
    consensus_distance = compute_consensus_distance(clients)
    print(f"Consensus distance: {consensus_distance:.4f}")
    consensus_distance_honest = compute_consensus_distance([c for i,c in enumerate(clients) if i >= num_malicious])
    print(f"Consensus distance (honest clients): {consensus_distance_honest:.4f}")
    
    # Test each client's model
    test_loader = torch.utils.data.DataLoader(dataset.get_test_dataset(), batch_size=32, shuffle=False)
    for client in clients:
        client.test(test_loader)

    # Print trust tracking for each client
    print("\n--- Trust Tracking Summary ---")
    for client in clients:
        if hasattr(client, 'trust_history'):
            print(f"Client {client.id} trust history:")
            for peer_id, scores in client.trust_history.items():
                print(f"  Peer {peer_id}: {[f'{s:.4f}' for s in scores]}")
    print("Simulation complete")

