# ECE60872-FLTrust-Eval

## Environment
This simulation uses Python 3.10 and PyTorch. See [environment.yml](environment.yml) for more specific information.
## How to Run the P2PFL Simulation
[simulate_P2PFL.py](simulate_P2PFL.py) runs a federated learning simulation that uses our novel Peer-to-Peer Federated Learning (P2PFL) approach that incorporates FL Trust with a basic CNN model. We also added a novel penalty boost to clients with continuous low trust scores during training which can be activated as described in the command below. 
The script can be run without any arguments and will set the number of clients to 5, number of malicious clients to 0, number of rounds to 5, and the trust score penalty boost set inactive by default.

Alternatively, these parameters are configurable. For example, this command runs the simulation with 6 total clients, 1 malicious client, for 10 rounds with the trust score penalty boost activated.
```
python simulate_P2PFL.py 6 1 10 true
```

The simulation will indicate trust scores for each client over time, each client's local model accuracy after training, and the final aggregated model accuracy.



