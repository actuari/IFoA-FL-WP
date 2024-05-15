
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
import run_config
import architecture as archit
import torch
import utils
import insur_maskedAgg
import platform

"""
This module demonstrates a Federated Learning (FL) server setup using the Flower framework.
It defines a custom aggregation strategy for metrics and initializes the server with specified configurations.
"""

def init_parameters() -> fl.common.Parameters:
    """
    Initialize model parameters using Xavier initialization and return them in a format suitable for FL.

    Returns:
        fl.common.ParametersRes: The model's parameters formatted for the Flower FL framework.
    """
    model = archit.MultipleRegression(num_features=run_config.NUM_FEATURES, num_units_1=run_config.NUM_UNITS_1, num_units_2=run_config.NUM_UNITS_2)
    params = archit.get_parameters(model)
    return fl.common.ndarrays_to_parameters(params)



def start_FL_server():
    """
    Starts a Federated Learning (FL) server with a custom aggregation strategy.

    This function initializes the random seed for reproducibility, sets up a custom strategy for aggregating updates from clients participating in the federated learning process, and starts the FL server. The server listens for client connections and orchestrates the federated learning process based on the specified strategy and server configuration.

    The aggregation strategy, `LocalUpdatesStrategy`, is configured to require participation from all clients for both training (fit) and evaluation phases. It also initializes the model parameters for the federated learning process.

    The server configuration specifies the number of federated learning rounds to be conducted.

    External Dependencies:
        - `utils.seed_torch`: Ensures reproducible results by setting the random seed.
        - `insur_secAgg.LocalUpdatesStrategy`: A custom class for the aggregation strategy.
        - `fl.server.start_server`: Function to initiate the FL server.
        - `run_config.server_config`: Contains server-side configurations such as `num_clients` and `num_rounds`.
        - `init_parameters()`: Function to initialize model parameters for the federated learning process.

    Note:
        The server address is hardcoded to "[::]:8080", indicating that the server will listen on all available interfaces and the port 8080.

    Example:
        To start the FL server, simply call this function from your script:
            ```python3 start_FL_server()```
    """
    utils.seed_torch()

    strategy = insur_maskedAgg.LocalUpdatesStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=run_config.server_config["num_clients"],
        min_evaluate_clients=1,
        min_available_clients=run_config.server_config["num_clients"],
        initial_parameters=init_parameters(),
    )


    system = platform.system()
    if system == "Windows":
        server_address="localhost:8080"
    else:
        server_address= "[::]:8080"


    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=run_config.server_config["num_rounds"]),
        strategy=strategy
    )


if __name__ == "__main__":
    start_FL_server()