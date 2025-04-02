# ECE60872-FLTrust-Eval

## How to Run the Simulation
[simulate_FedAvg.py](simulate_FedAvg.py) runs a federated learning simulation that uses FedAvg as the aggregation rule with a basic CNN model. The FedAvg simulation serves as the baseline for our experiments since it does not have any defenses.
The script can be run without any arguments and will set the number of clients to 5, number of malicious clients to 0, and number of rounds to 3 by default. Note: the current state of the malicious client's attack simply flips the sign of the loss value during training.

Alternatively, these parameters are configurable. For example, this command runs the simulation with 10 total clients, 1 malicious client, for 10 rounds.
```
python simulate_FedAvg.py 6 1 10
```

The simulation will indicate overall test accuracy of the global model.

## Future Work
Next steps include the following:
- Implement FLTrust, Krum, and Trimmed Mean/Median to compare against the baseline FedAvg simulation
- Implement the following attacks: label flipping, Krum/Trim, Scaling
- Run experiments against the established defenses with our attacks to identify vulnerabilities
- Design our own novel, byzantine-robust aggregation rule to evaluate against FLTrust based on our experiments

