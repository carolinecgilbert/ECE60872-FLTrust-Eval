from client import *
from server import Server
from dataset import MNISTDataset
from model import CNN

import torch
import sys
import time


if __name__ == "__main__":
    # Default client set up
    num_clients = 5
    num_malicious = 0
    rounds = 3

    # Check for command line arguments to change client setup
    if len(sys.argv) > 1:
        num_clients = max(int(sys.argv[1]), 1)
    if len(sys.argv) > 2:
        num_malicious = min(int(sys.argv[2]), num_clients)
    if len(sys.argv) > 3:
        rounds = max(int(sys.argv[3]), 1)
    print(f"Number of clients: {num_clients}")
    print(f"Number of malicious clients: {num_malicious}")
    print(f"Number of rounds: {rounds}")

    # Initialize the global model and dataset
    global_model = CNN()
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

    # Visualize sample images from the training dataset
    dataset.visualize_samples()

    # Initialize clients
    clients = []
    for i, idxs in enumerate(client_indices):
        if i < num_malicious:
            clients.append(SignFlipping(i, CNN(), train_dataset, idxs, device))
        else:
            clients.append(Client(i, CNN(), train_dataset, idxs, device))

    # Initialize the server
    server = Server(global_model, device)

    # Simulate federated learning rounds
    print("Starting federated learning simulation...")
    start_time = time.time()
    for r in range(rounds):
        client_weights = []
        for client in clients:
            client.model.load_state_dict(server.global_model.state_dict())
            weights = client.train(epochs=1)
            client_weights.append(weights)

        server.aggregate(client_weights)
        print(f"Round {r+1} complete")
    end_time = time.time()
    print(f"Federated learning simulation completed in {end_time - start_time:.2f} seconds")

    # Test the global model
    test_loader = torch.utils.data.DataLoader(dataset.get_test_dataset(), batch_size=32, shuffle=False)
    server.test_global_model(test_loader)
    print("Simulation complete")