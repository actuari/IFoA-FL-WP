import numpy as np
import torch
from torch import nn
from typing import List


class MultipleRegression(nn.Module):
    """
    A neural network model for multiple regression with two hidden layers.

    Attributes:
        layer_1 : nn.Linear
            The first linear layer.
        layer_2 : nn.Linear 
            The second linear layer.
        layer_out : nn.Linear 
            The output layer.
        dropout : nn.Dropout 
            Dropout layer for regularization.
        act : activation
            The activation function.

    Parameters:
        num_features :  int
            Number of input features. Default is 39.
        num_units_1 : int 
            Number of units in the first hidden layer. Default is 25.
        num_units_2 : int
            Number of units in the second hidden layer. Default is 2.
        activation : callable 
            Activation function to use. Default is nn.Tanh.
        dropout_rate : float
            Dropout rate. Default is 0.

    Methods:
        forward(inputs) 
            Performs a forward pass through the model.
        predict(test_inputs)
            Predicts outputs for the given inputs.
    """
    
    def __init__(self, num_features=39, num_units_1=25, num_units_2=2, activation=nn.Tanh, dropout_rate=0):
        super(MultipleRegression, self).__init__()
        self.layer_1 = nn.Linear(num_features, num_units_1)
        self.layer_2 = nn.Linear(num_units_1, num_units_2)
        self.layer_out = nn.Linear(num_units_2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = activation()

        # Initialize weights and biases
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.zeros_(self.layer_1.bias)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.zeros_(self.layer_2.bias)
        torch.nn.init.xavier_uniform_(self.layer_out.weight)
        torch.nn.init.zeros_(self.layer_out.bias)
    
    def forward(self, inputs):
        """
        Performs a forward pass through the model.

        Parameters:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: The model's output tensor.
        """
        x = self.dropout(self.act(self.layer_1(inputs)))
        x = self.dropout(self.act(self.layer_2(x)))
        x = torch.exp(self.layer_out(x))
        return x

    def predict(self, test_inputs):
        """
        Predicts outputs for the given inputs without applying dropout.

        Parameters:
            test_inputs (Tensor): Input tensor for prediction.

        Returns:
            Tensor: The predicted output tensor.
        """
        x = self.act(self.layer_1(test_inputs))
        x = self.act(self.layer_2(x))
        x = torch.exp(self.layer_out(x))
        return x

def get_parameters(model) -> List[np.ndarray]:
    """
    Retrieves the parameters of the given model as a list of numpy arrays.

    Parameters:
        model (nn.Module): The model from which to retrieve parameters.

    Returns:
        List[np.ndarray]: A list containing the model's parameters as numpy arrays.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
