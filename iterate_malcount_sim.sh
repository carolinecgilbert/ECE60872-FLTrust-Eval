#!/bin/bash

# Number of total workers
TOTAL_WORKERS=10
ROUNDS=5

rm logs/*

# Range of malicious workers (0 to 10 in this case)
for i in {0..10}
do
    echo "Running simulation with $i malicious workers..."
    ~/pyenv/bin/python3 simulate_FedAvg.py $TOTAL_WORKERS $i $ROUNDS > logs/run_${i}_malicious.log
done

echo "Done."