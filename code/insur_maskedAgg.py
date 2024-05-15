
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from functools import reduce
from utils_quantisation import quantize, dequantize_mean, add_mod, dequantize, CR, N 
from run_config import QUANTISATION
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy.strategy import Strategy
from flwr.common.typing import Status
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    NDArray,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from logging import WARNING


def aggregate_weighted_average(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """
    Compute the weighted average of model parameters.

    Parameters:
        results (List[Tuple[NDArrays, int]]): A list of tuples where each tuple contains model parameters (NDArrays) and the number of examples (int) used for training.

    Returns:
        NDArrays: The weighted average of model parameters.
    """
    num_examples_total = sum([num_examples for _, num_examples in results])
#    print('num examples total:')
#    print(num_examples_total)

    weighted_weights = [
        [layer.astype(np.float64) * np.float64(num_examples) for layer in weights] for weights, num_examples in results
    ]

#    print('agent0: ')
#    print(weighted_weights[0][0])
#    print('agent1:')
#    print(weighted_weights[1][0])

    weights_avg: NDArrays = [
        (reduce(np.add, layer_updates) / np.float64(num_examples_total)).astype(np.float64)
        for layer_updates in zip(*weighted_weights)
    ]

#    print('weights_prime')
#    print(weights_prime)

    return weights_avg


def aggregate_qt(results: List[Tuple[NDArrays, np.int64]]) -> NDArrays:
    """
    Compute the weighted average of quantized model parameters and dequantize the result.

    Parameters:
        results (List[Tuple[NDArrays, int]]): A list of tuples containing quantized model parameters (NDArrays)  and the number of examples (int) used for training.

    Returns:
        NDArrays: The weighted and dequantized average of model parameters.
    """

    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    print('num examples total:')
    print(num_examples_total)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    print('agent0: ')
    print(weighted_weights[0][0])
    print('agent1:')
    print(weighted_weights[1][0])
     
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        (reduce(add_mod, layer_updates))
        for layer_updates in zip(*weighted_weights)
    ]

    #dequantize:
    print('num_examples_total', num_examples_total)
    weights_deq = dequantize(weights_prime, CR, N, num_examples_total)
    weights_deq_weigh = np.divide(weights_deq, num_examples_total)

    print('weights_deq_weigh', weights_deq_weigh)

    return weights_deq_weigh


def aggregate_qt2(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    print('num examples total:')
    print(num_examples_total)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    print('type(weighted_weitgts)', type(weighted_weights))
    print('type(weighted_weitgts[0])', type(weighted_weights[0]))
    print('type(weighted_weitgts[0][0])', type(weighted_weights[0][0]))

    print('agent0: ')
    print(weighted_weights[0][0])
    print('agent1:')
    print(weighted_weights[1][0])
     
    # Compute average weights of each layer
    weights_prime: NDArrays = [
        (reduce(add_mod, layer_updates))
        for layer_updates in zip(*weighted_weights)
    ]

    #dequantize:
    print('num_examples_total', num_examples_total)
    weights_deq_weigh = dequantize_mean(weights_prime, CR, N, num_examples_total)
   
    print('weights_deq_weigh', weights_deq_weigh)

    return weights_deq_weigh



class LocalUpdatesStrategy(fl.server.strategy.FedAvg):
    """
    A strategy for federated averaging that considers local updates with optional quantization.

    Extends the FedAvg strategy from Flower to aggregate fit results using custom logic, 
    including support for quantized model parameter aggregation.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """
        Aggregate fit results using weighted average with support for quantized parameters.

        Parameters:
            server_round (int): The current round of the server.
            results (List[Tuple[ClientProxy, FitRes]]): The results from clients' local training.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): A list of clients that failed to return results.

        Returns:
            Tuple[Optional[Parameters], Dict[str, float]]: Aggregated parameters and aggregated metrics.
        """


        if not results or (not self.accept_failures and failures):
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results # ms test 10.09.2023 prev: (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['exposure']) for _, fit_res in results
        ]

        if QUANTISATION:
            parameters_aggregated = ndarrays_to_parameters(aggregate_qt2(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(aggregate_weighted_average(weights_results))
            
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")


        return parameters_aggregated, metrics_aggregated
