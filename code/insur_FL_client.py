

import utils
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import flwr as fl 
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import architecture as archit
import copy
import csv
from utils_quantisation import quantize, modulus,  N, M, CR
from utils_smpc import calc_noise, calc_noise_zero
import run_config
from run_config import EPOCHS, BATCH_SIZE, QUANTISATION, SMPC_NOISE, LEARNING_RATE

MODEL_PATH = '.'
ROUND_NO = 0

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class ClaimsFrequencyFLClient(fl.client.NumPyClient):
    """
    A client class for federated learning using Flower framework, designed to handle the training and evaluation 
    of a machine learning model on local data and interact with a federated learning server.

    Attributes:
        - model (torch.nn.Module): The local machine learning model.
        - optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        - criterion (torch.nn.Module): The loss function used for model training.
        - trainset (torch.utils.data.Dataset): The training dataset.
        - valset (torch.utils.data.Dataset): The validation dataset.
        - testset (torch.utils.data.Dataset): The test dataset.
        - num_examples (Dict): A dictionary containing the number of examples.
        - exposure (float): A parameter used to weight the parameter updates.
        - noise (List): A list of noise values added to model parameters for secure aggregation. 
    """
    
    def __init__(
        self,
        model,#: archit.MultipleRegression(num_features=run_config.NUM_FEATURES, num_units_1=run_config.NUM_UNITS_1, num_units_2=run_config.NUM_UNITS_2),  #archit.NeuralNetworks(NUM_FEATURES), # type: ignore
        optimizer, 
        criterion,
        trainset: torch.utils.data.dataset,
        valset: torch.utils.data.dataset,
        testset: torch.utils.data.dataset,
        num_examples: Dict,
        exposure: float,
        noise: List,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.num_examples = num_examples
        self.exposure = round(exposure) # we round it to int as we will use it to weight the parameter updates
        self.stats = [] 
        self.noise = noise 


    def get_parameters(self, config) -> List[np.ndarray]:
        """
            Retrieves local model parameters, with optional quantization and noise addition based on configuration flags.

            This method extracts the model's parameters as numpy arrays. If the QUANTISATION flag is set in the provided
            configuration, the parameters are quantized. Similarly, if the SMPC_NOISE flag is set, noise is added to the
            parameters. This is part of preparing the model's parameters for secure multi-party computation (SMPC) or
            other privacy-preserving mechanisms.

            Parameters:
                config : dict  
                    A configuration dictionary that may contain flags like QUANTISATION and SMPC_NOISE to indicate whether quantization or noise addition should be applied to the model parameters.

            Returns:
                List[np.ndarray]
                    A list of numpy arrays representing the model's parameters after applying quantization and/or noise addition as specified in the configuration. Each numpy array in the list corresponds to parameters of a different layer or component of the model.

            Examples:
                >>> model = YourModelClass()
                >>> config = {'QUANTISATION': True, 'SMPC_NOISE': False}
                >>> parameters = model.get_parameters(config)
                >>> type(parameters)
                <class 'list'>
                >>> type(parameters[0])
                <class 'numpy.ndarray'>

            Notes:
                - QUANTISATION and SMPC_NOISE are flags handling quantization and noise addition.

            """


        self.model.train()
        #st_dict_np = [np.round(val.cpu().numpy(),3) for _, val in self.model.state_dict().items()]
        #st_dict_np = [val.cpu().numpy().astype(np.float64) for _, val in self.model.state_dict().items()]
        #st_dict_np = [np.round(val.cpu().numpy(),5) for _, val in self.model.state_dict().items()]
        st_dict_np = [val.cpu().numpy().astype(np.float_) for _, val in self.model.state_dict().items()]
        
        global ROUND_NO

        if QUANTISATION:
            st_dict_np = quantize(st_dict_np, CR, N)
            k = 0
            for i in range(len(st_dict_np)):
                for j in range(len(st_dict_np[i])):
                    st_dict_np[i][j] = modulus(st_dict_np[i][j] + self.noise[ROUND_NO][k])         
                    k = k + 1
        elif SMPC_NOISE:
            k = 0
            for i in range(len(st_dict_np)):
                for j in range(len(st_dict_np[i])):
                    st_dict_np[i][j] += np.float64(self.noise[ROUND_NO][k])     
                    k = k + 1

        ROUND_NO = ROUND_NO + 1

        print('Round' + str(ROUND_NO))

        return st_dict_np 

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Updates the model's parameters with new values provided as a list of NumPy ndarrays.

        This method takes a list of NumPy arrays containing new parameter values and updates the model's
        parameters accordingly. It's typically used to set model parameters after they have been modified
        or updated elsewhere, possibly after aggregation in a federated learning scenario or after receiving
        updates from an optimization process.

        Parameters:
            parameters : List[np.ndarray]
                A list of NumPy ndarrays where each array corresponds to the parameters for a different layer or
                component of the model. The order of the arrays in the list should match the order of parameters
                in the model's state_dict.

        Returns:
            None

        Examples:
            >>> model = YourModelClass()
            >>> new_parameters = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.5, 0.6])]
            >>> model.set_parameters(new_parameters)
            >>> # Model parameters are now updated with `new_parameters`.

        Notes:
            - This method assumes that the provided list of parameters matches the structure and order of the model's parameters. If the order or structure of `parameters` does not match, this may lead to incorrect assignment of parameters or runtime errors.
            - The method converts each NumPy ndarray to a PyTorch tensor before updating the model's state dict. Ensure that the data types and device (CPU/GPU) of the NumPy arrays are compatible with your model's requirements.
        """

        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int,  Dict]:
        """
        Trains the model locally using provided parameters and configuration settings.

        This method sets the initial model parameters, trains the model on a local dataset,
        and returns the updated model parameters after training. It encapsulates the process of
        local training within a federated learning framework, including setting initial parameters,
        executing the training loop, and optionally evaluating the model on a validation set.

        Parameters:
            parameters : List[np.ndarray]
                A list of NumPy ndarrays representing the model parameters to be set before training begins.
                These parameters might come from a central server in a federated learning setup.

            config : Dict[str, str]
                A dictionary containing configuration options for the training process. This may include
                hyperparameters such as learning rate, batch size, or any other model-specific settings.

        Returns:
            Tuple[List[np.ndarray], int, Dict]
                A tuple containing three elements:
                    - A list of NumPy ndarrays representing the updated model parameters after training.
                    - An integer representing the number of training samples used in the training process. This could be used for weighted averaging in a federated learning setup.
                    - A dictionary containing additional information about the training process. For example, it could include metrics such as training loss or accuracy, or model-specific metrics like 'exposure' in this case.

        Examples:
            >>> model_trainer = ModelTrainer(model, optimizer, criterion, trainset, valset, testset)
            >>> updated_params, num_samples, metrics = model_trainer.fit(initial_params, config)
            >>> print(metrics)
            {'exposure': ...}

        Notes:
            - The training process uses the DataLoader from PyTorch to load the training and validation datasets with the specified batch size. It's important to ensure that the datasets are properly initialized and passed to the ModelTrainer before calling `fit`.
            - The configuration dictionary must include all necessary settings required by the training and evaluation process. Missing configurations might result in default values being used or in runtime errors.
            - The method internally calls `set_parameters` to set the model's initial parameters and `get_parameters` to retrieve the updated parameters after training. Ensure that these methods are implemented correctly for the `fit` method to work as expected.
            - This method appends the training statistics to an internal list `self.stats` after each training session, allowing for tracking of performance over multiple rounds of federated learning.

        """

        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        trainLoader = DataLoader(self.trainset, batch_size=BATCH_SIZE, shuffle=True)
        valLoader = DataLoader(self.valset, batch_size=BATCH_SIZE)
        _, loss_stat = train(self.model, self.optimizer, self.criterion, trainLoader, valLoader, epochs=EPOCHS )
        self.stats.append(loss_stat)   # after FL local training append learning curves

        # test predictions on test dataset:
        testloader = DataLoader(self.testset, batch_size=BATCH_SIZE)
        loss = test(self.model, self.criterion, testloader)
        
        return self.get_parameters(config), 1, {'exposure': self.exposure}  # ms to test now 10.09.2023 prev :self.get_parameters(config), len(self.trainset), {'exposure': self.exposure}


    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluates the model with the provided global parameters on local test data.

        This method is intended to be used in a federated learning context where global model parameters are evaluated
        on a client's local test dataset. The method sets the model's parameters to the provided global parameters, evaluates
        these parameters on the local test dataset, and returns the evaluation loss, the number of test samples, and a dictionary
        containing evaluation metrics such as accuracy.

        Parameters:
            - parameters : List[np.ndarray]
                A list of NumPy ndarrays representing the global model parameters to be evaluated.
            - config : Dict[str, str]
                A dictionary containing configuration options for the evaluation process. This could include model-specific settings or evaluation hyperparameters. Note: Currently, this parameter is not directly used in the method but is included for consistency and future extensions.

        Returns:
            Tuple[float, int, Dict]
                A tuple containing three elements:
                    - The evaluation loss as a float.
                    - The number of samples in the test dataset as an int.
                    - A dictionary containing evaluation metrics, with at least an "accuracy" key providing the accuracy of the model on the test dataset calculated as the loss divided by the number of test loader batches.

        Examples:
            >>> client_evaluator = ModelEvaluator(model, criterion, testset)
            >>> global_params = [...]  # Global parameters obtained from the server
            >>> loss, num_samples, metrics = client_evaluator.evaluate(global_params, {})
            >>> print(f"Test Loss: {loss}, Test Accuracy: {metrics['accuracy']}")

        Notes:
            - The method utilizes a DataLoader to iterate through the test dataset, and the batch size for the DataLoader is determined by the global BATCH_SIZE variable.\n
            - The accuracy calculation in the returned dictionary is a simplified example. Depending on the model and the task, you might need a more sophisticated method to calculate accuracy or other relevant metrics.\n
            - Ensure that the global `BATCH_SIZE` variable is appropriately set for the evaluation DataLoader to function correctly.

        """
        
        self.set_parameters(parameters)
    
        testloader = DataLoader(self.testset, batch_size=BATCH_SIZE)
        loss = test(self.model, self.criterion, testloader)

        return float(loss), len(self.testset), {"loss": loss/len(testloader)}


def train(
        model, 
        optimizer, 
        criterion,
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        epochs=EPOCHS
    ) :
    """
    Trains the given model using the specified training and validation data loaders, optimizer, and loss function
    across a defined number of epochs. Evaluates the model on the validation dataset after each training epoch and
    reports the training and validation losses.

    Parameters:
        model : torch.nn.Module
                The neural network model to be trained.
        optimizer : torch.optim.Optimizer
            The optimizer used for adjusting the model parameters based on the computed gradients.
        criterion : torch.nn.Module
            The loss function used to evaluate the goodness of the model's predictions.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training data, providing batches of data.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation data, used to assess the model's performance.
        epochs : int, optional
            The number of complete passes through the training dataset. Defaults to the global variable `EPOCHS`.

    Returns:
        Tuple[torch.nn.Module, Dict[str, List[float]]]
            A tuple containing:
            - The trained model.
            - A dictionary with keys 'train' and 'val', each mapping to a list of loss values recorded at the end of each epoch.

    Example:
        >>> model = MyCustomModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        >>> val_loader = DataLoader(val_dataset, batch_size=64)
        >>> trained_model, loss_stats = train(model, optimizer, criterion, train_loader, val_loader, epochs=10)
        >>> print(loss_stats['train'])
        >>> print(loss_stats['val'])

    Notes:
        - This function assumes that the `model`, `optimizer`, `criterion`, and data loaders have been initialized
        before being passed as arguments.
        - The function sets the model in training mode (`model.train()`) at the beginning of each epoch and uses the optimizer
        to update model parameters based on the computed gradients.
        - Losses for both training and validation phases are accumulated over each epoch and reported at the end.
        - It's important to ensure that the device (`cpu` or `cuda`) is correctly configured for the model, data, and criterion
        before calling this function.
    """
    
    loss_stats = {
                    'train': [],
                    'val': []
                 }

    for e in range(epochs):
        train_epoch_loss = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            
        val_epoch_loss = test(model, criterion,  val_loader)

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))                              
        
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')

    return model, loss_stats


def test(model, criterion, val_loader)-> float:
    """
    Evaluates the performance of the model on a validation dataset.

    This function iterates over the provided validation DataLoader, computes the loss of the model predictions
    against the true labels using the provided loss criterion, and sums up the loss over all validation batches
    to get the total validation loss. The model is set to evaluation mode during this process to disable dropout
    or batch normalization layers that behave differently during training.

    Parameters:
        model : torch.nn.Module
            The neural network model to be evaluated. It should already be trained or loaded with pre-trained weights.          
        criterion : torch.nn.Module
            The loss function used to calculate the loss between the model predictions and the true labels.     
        val_loader : torch.utils.data.DataLoader
            A DataLoader providing batches of validation data including features and labels.

    Returns:
        float
            The total loss computed over all batches of the validation dataset.

    Example:
        >>> model = MyCustomModel()
        >>> criterion = torch.nn.MSELoss()
        >>> val_dataset = CustomDataset(...)
        >>> val_loader = DataLoader(val_dataset, batch_size=64)
        >>> total_val_loss = test(model, criterion, val_loader)
        >>> print(f'Total Validation Loss: {total_val_loss}')

    Notes:
        - The function automatically moves the input and target data to the same device as the model before making predictions.
        Ensure that the model and criterion are already moved to the appropriate device (CPU or GPU) before calling this function.
        - The function uses `torch.no_grad()` context manager to disable gradient computation during evaluation, improving memory
        efficiency and speed.
        - It's important to call `model.eval()` before evaluating the model to set the model to evaluation mode. This is necessary
        for models that have layers like dropout or batch normalization that behave differently during training and evaluation.
    """
    

    with torch.no_grad():
        val_epoch_loss = 0   
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))      
            val_epoch_loss += val_loss.item()
    return val_epoch_loss


def parse_args():
    """
    Parses command-line arguments for training configuration.

    This function uses `argparse` to define and parse command-line arguments necessary for specifying the training mode (centralized or federated learning), the number of federated learning participants, and the partition of data to use for training.

    Returns:
        Namespace: An argparse.Namespace object containing the arguments and their values. 
        - `agent_id` (int): Specifies the partition of data for training. A value of -1 indicates that all data should be used ( training a global model ). Valid values are in the range [-1, run_config.server_config["num_clients"]].
        - `if_FL` (int): Determines the pipeline type. A value of 0 sets the training to centralized (currently disabled), while a value of 1 sets it to federated learning.

    Example command line usage:
        For federated learning with all data:
        `python script.py --agent_id -1 --if_FL 1`

        For federated learning for agent 3:
        `python script.py --agent_id 3 --if_FL 1`
    """
    parser = argparse.ArgumentParser(description="Run federated or centralized training")
    parser.add_argument("--agent_id", type=int, default=-1, choices=range(-1, 10),
                        help="Partition of data for training; -1 for all data")
    parser.add_argument("--if_FL", type=int, default=1, choices=[0, 1],
                        help="Pipeline type: 0 for centralized, 1 for federated learning")
    return parser.parse_args()


def initialize_model():
    """
    Initializes and returns the machine learning model along with the optimizer and loss criterion.

    This function initializes a multiple regression model with architecture specified by 
    `archit.MultipleRegression`. The model configuration (number of features and units in the first and second layer)
    is read from `run_config`. The model is moved to a device (e.g., CPU or GPU) specified by the global `device` variable.
    The optimizer used is NAdam, and the loss criterion is Poisson negative log likelihood.

    Note:
        - The function relies on global variables `device`, `run_config`, `archit`, `optim`, and `nn` 
          for its operation.
        - `run_config` should have `NUM_FEATURES`, `NUM_UNITS_1`, and `NUM_UNITS_2` attributes defined.
        - `device` should be defined globally and indicate where the model should be allocated (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing three elements:
            - model (archit.MultipleRegression): The initialized multiple regression model.
            - optimizer (optim.NAdam): The optimizer for the model.
            - criterion (nn.PoissonNLLLoss): The loss criterion.

    Example:
        >>> model, optimizer, criterion = initialize_model()
        >>> print(type(model))
        <class 'archit.MultipleRegression'>
        >>> print(type(optimizer))
        <class 'torch.optim.nadam.NAdam'>
        >>> print(type(criterion))
        <class 'torch.nn.modules.loss.PoissonNLLLoss'>
    """

    model = archit.MultipleRegression(
        num_features=run_config.NUM_FEATURES,
        num_units_1=run_config.NUM_UNITS_1,
        num_units_2=run_config.NUM_UNITS_2
    ).to(device)
    optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.PoissonNLLLoss(log_input=False, full=True)
    return model, optimizer, criterion


def main():
    """
    Main function to execute the training process.

    This function begins by parsing command-line arguments to configure the training session. It then initializes the random seeds for reproducibility and processes the specified client's dataset. Based on the provided arguments, it either proceeds with federated learning or exits if centralized training is indicated.

    Federated learning involves initializing a model, optimizer, and loss criterion, followed by applying noise to the training process for privacy preservation. The model is then trained with data from the specified client, communicating with a federated learning server as configured.

    Finally, the trained model's state is saved, and training loss statistics are written to a file.

    Returns:
        int: 0, indicating successful completion of the training process.

    Note:
        This function relies on external configurations from `run_config`, utility functions from `utils`, and the `ClaimsFrequencyFLClient` for the FL training setup. It assumes the presence of a federated learning server listening on the specified address.
    
    Example usage:
        To run this script, ensure that the command-line arguments are correctly set for the desired training configuration. For example:
        `python3 insur_FL_client.py --agent_id 1 --if_FL 1`
        This would initiate federated learning for client with agent_id 1.
    """

    args = parse_args()
    utils.seed_torch() 
    
    print(f'Processing client {args.agent_id}')
    train_dataset, val_dataset, test_dataset, _, _, exposure = utils.load_individual_data(args.agent_id, run_config.IF_TRAIN_VAL)  # in folder data each training participant is storing their private, unique dataset 

    model, optimizer, criterion = initialize_model()

    if not args.if_FL:
        return 0  # If not federated learning, exit early
    
    noise_func = calc_noise if SMPC_NOISE else calc_noise_zero
    noise = noise_func('../data/seeds_250.csv', args.agent_id)
    
    
    model_name = f'fl_model_{run_config.run_name}.pt'
    agent_path = f'../ag_{args.agent_id}/{model_name}' if args.agent_id in range(10) else f'../ag_-1/{model_name}'


    agent = ClaimsFrequencyFLClient(model, optimizer, criterion, train_dataset, val_dataset, test_dataset, {}, exposure, noise)
    fl.client.start_numpy_client(server_address="[::]:8080", client=agent,)     # 

    torch.save(model.state_dict(), agent_path)      

    # Writing loss stats to file
    with open(f'../ag_{args.agent_id}/los_stats.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['train', 'val'])
        writer.writeheader()
        writer.writerows(agent.stats)

    print("FL training completed; loss_results saved!")

    return 0

if __name__ == "__main__":
          main()
