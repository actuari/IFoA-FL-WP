# IFoA Federated Learning Working Party
Author list:
Malgorzata Smietanka (Chair), 
Dylan Liew (Deputy), 
Scott Patric,
Harry Lohhy


# Introduction

This project has been developed by the IFoA Federated Learning Working Party to demonstrate the application of Federated Learning (FL) in the insurance use case. 
The project is the basis for this paper: Privacy Preserving Neural Network Predictive Modelling in Insurance using Federated Learning, available on SSRN. 

# Dependency Management

We recommend that you use Poetry to install dependencies and manage your virtual environment (Poetry installation: https://python-poetry.org/docs/#installing-with-the-official-installer ), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

**RUN** `poetry install`

Activate the poetry environment 

**RUN** `poetry shell`

**RUN** `python3 -c "import flwr"`
If you don't see any errors you're good to go!

# Project structure 

IFoA-FL-WP
    poetry.lock
    pyproject.toml
    ag_-1
    ag_0
    ag_1
    ag_2
    ...
    ag_9
    code
    results
    docs
    data

poetry.lock, pyproject.toml - files containing the requirements and dependencies for setting up an environment with Poetry
ag_0, ... ag_9  - folders representing the FL participating companies.Each folder is unique and private to the company and is used to store partial models for local model training. After FL training, the final model is stored in each participant's folder.  
ag_-1 - folder dedicated to the Global Model scenario. 
code -  main use case code. It includes both server and client code as well as diagnostic notebooks ( results_graph_notebook.ipynb), hyperparameter tuning code ( skorch_tuning.py).
results - folder containing results of hyperparameter tuning with code/skorch_tuning.py 

# Starting Federated Learning training  :

**0. navigate to  /code folder**

**1. Setup config in run_config.py** 
In this file you set up the Federated Learning run configuration to define the number of agents/clients collaborating on the Federated Learning model. 

In this use case we use 10 clients
   server_config = {
    "num_clients": 10,
     .... 
   }

**2. Prepare dataset for each individual agent** 

**RUN** `python3 prepare_dataset.py`

This creates 10 uniformly split training and validation files (including 1 Test set)

Optional step: Check if split has been executed correctly:

**RUN** `python3 check_dataset.py --agents=10`

**3. Tune Gloabal Model and 10 Local Models**

**RUN** `python3 skorch_tuning.py`

This should output and save all models in .pkl as well as training loss curves, and hyperparameter tuning results.

**NOTE** You will need folders for ag_-1, ag_0,...,ag_9 in your repo.

**4. Get best hyperparameter results (for Federated Model)**

Take hyperparameters from 1st row of 'local_average_HPT_performance.csv' in `/results` (which should be automatically output from step above)

Note the format of the `params_key` is `learning_rate_layer1neurons_layer2neurons_bath_size` i.e. 3 tuned hyperparameters. Also note on my machine this is currently `0.001_60_20_100` (could change if run on different machine with different seed, OS etc. but should be reproducible on each machine)

update the below parameters in run_config.py based hyperparameter results
BATCH_SIZE = 
LEARNING_RATE = 
NUM_UNITS_1 =  
NUM_UNITS_2 =  


**5. Start Federated Learning server:**

**RUN** `python3 IFoA_server.py`

**7. Start Federated Learning clients:**

To train the Federated Learning model together with 10 agents, you need to open 10 separate terminal windows and run client code:
for client number i. Note that in our use case i is in range(0,10).

**RUN** `python3 'IFoA_client.py' --agent_id=i `

(You might consider writing a bash script to automate this process).


**8. Output model performance**

**RUN**: `results_graph_notebook.ipynb`


# Citation
If you publish work based on this use case, please cite it as follows:

@article{IFoAFLWP2024,
  title={Privacy Preserving Neural Network Predictive Modelling in Insurance using Federated Learning},
  author={Smietanka, Malgorzata and Liew, Dylan and Patric, Scott and Lohhy, Harry},
  journal={SSRN preprint},
  year={2024}
}