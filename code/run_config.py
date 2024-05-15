from typing import Dict
import pandas as pd

# Constants and Configuration
IF_TRAIN_VAL = 0  # Include validation dataset in training
QUANTISATION = 0
SMPC_NOISE = 1  
EPOCHS = 10
BATCH_SIZE = 500
NUM_FEATURES = 39
LEARNING_RATE = 0.001
NUM_UNITS_1 = 15 # Extracting top results for NUM_UNITS_1 and NUM_UNITS_2
NUM_UNITS_2 = 5 

# Server Configuration
server_config = {
    "num_clients": 10,
    "num_rounds": 350
}


# Dataset Configuration
dataset_config = {
    "path": '../data/freMTPL2freq.csv',
    "seed": 300,
    "num_agents": server_config["num_clients"],
    "num_features": NUM_FEATURES
}

DATA_PATH = dataset_config["path"]
SEED = dataset_config["seed"]


# Constructing a unique run name based on configuration
run_name = f'RndTuning_{server_config["num_clients"]}ag_{server_config["num_rounds"]}rnd_{EPOCHS}ep_{QUANTISATION}qt_{SMPC_NOISE}SMPCn'



if __name__ == "__main__":
    print(f"Run name: {run_name}")
    print(f"Top tuning results for NUM_UNITS_1: {NUM_UNITS_1}, NUM_UNITS_2: {NUM_UNITS_2}")
