import torch as th
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from run_config import NUM_FEATURES, DATA_PATH, SEED
import architecture
from skorch import NeuralNetRegressor

#---------------------- DATASET CHECKS: ----------------------------------------------------------------------------

def row_check(agents: int = 10):
    """
    Validates the integrity of the dataset across multiple agents by checking the total number of rows, 
    the sum of exposure values, and the sum of claims across training, validation, and test datasets.

    This function ensures that the combined dataset from multiple agents, along with the test dataset,
    matches expected values for total row count, total exposure, and total claims. These checks are critical for
    verifying data integrity and consistency before proceeding with further data analysis or model training.

    Parameters:
        agents : int, optional
            The number of agents (or partitions) for which training and validation datasets are available.
            Defaults to 10.

    Raises:
        AssertionError
            If the total number of rows, total exposure, or total claims do not match expected values, 
            an AssertionError is raised indicating which specific integrity check has failed.

    Notes:
        - Assumes existence of CSV files in '../data/' following specific naming conventions.
        - Useful for data preprocessing in machine learning workflows involving multiple sources or agents.
        - 'Exposure' and '0' are assumed to be column names in the respective CSV files for exposure and claims.

    Example:
        >>> row_check(agents=5)
        # Checks datasets for 5 agents, along with the test dataset, and prints the status of each check.
    """
    # Helper function to load a CSV file into a DataFrame
    def load_data_frame(prefix: str, index: int) -> pd.DataFrame:
        return pd.read_csv(f'../data/{prefix}_{index}.csv')

    # Initialize expected values
    expected_row_count = 678_013
    expected_exposure_sum = 358_360.11
    expected_claims_sum = 36_056

    # Load and aggregate datasets
    datasets = {'X_train': [], 'X_val': [], 'y_tr': [], 'y_vl': []}
    for prefix in datasets.keys():
        for i in range(agents):
            datasets[prefix].append(load_data_frame(prefix, i))
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')

    # Calculate totals
    total_row_count = sum(df.shape[0] for prefix in ['X_train', 'X_val'] for df in datasets[prefix]) + X_test.shape[0]
    total_exposure_sum = sum(df['Exposure'].sum() for prefix in ['X_train', 'X_val'] for df in datasets[prefix]) + X_test['Exposure'].sum()
    total_claims_sum = sum(df.sum() for prefix in ['y_tr', 'y_vl'] for df in datasets[prefix]).item() + y_test.sum().item()

    # Validate dataset integrity
    assert total_row_count == expected_row_count, f"Total row count mismatch: expected {expected_row_count}, got {total_row_count}"
    assert round(total_exposure_sum, 2) == round(expected_exposure_sum, 2), f"Total exposure mismatch: expected {expected_exposure_sum}, got {round(total_exposure_sum, 2)}"
    assert total_claims_sum == expected_claims_sum, f"Total claims sum mismatch: expected {expected_claims_sum}, got {total_claims_sum}"
    
    print('All checks passed successfully.')


#---------------------------- SKORCH_TUNING_UTILS ------------------------------------------------------
def training_loss_curve(estimator, agent_id):
    """
    Plots the training and validation loss curves along with the percentage of Poisson Deviance Explained (PDE).

    Parameters:
        estimator : object
            The trained model or estimator that contains the training history. It is expected
            to have a 'history' attribute that is a NumPy array or a similar structure with
            'train_loss', 'valid_loss', and 'weighted_PDE_best' columns.
        agent_id : int or str
            Identifier for the agent. Used for titling the plot and naming the saved figure file.
    
    Notes:
        - This function saves the generated plot as a PNG file in a directory named after the agent.
        - Ensure the directory '../ag_{agent_id}/' exists or adjust the save path accordingly.
        - The function uses matplotlib for plotting and requires this library to be installed.
    """
    
    # Creating a DataFrame from the estimator's history for easier manipulation
    train_val_loss_df = pd.DataFrame(estimator.history[:, ['train_loss', 'valid_loss']], columns=['train_loss', 'valid_loss'])


    fig, ax = plt.subplots(figsize=(15,8))
    plt.plot(train_val_loss_df ['train_loss'],  label='Training Loss')
    plt.plot(train_val_loss_df ['valid_loss'],  label='Validation Loss')
    plt.legend(
        loc='upper right', 
        fontsize=15
        )
    plt.xlabel('Epochs', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.title(f"Agent {agent_id}'s Best Model's Training Loss Curve", fontsize=15)

    # Title and save figure
    plt.title(f"Agent {agent_id}'s Best Model's Training Loss Curve", fontsize=15)
    plt.savefig(f'../ag_{agent_id}/agent_{agent_id}_training_loss_chart.png', facecolor='white')
    plt.close(fig)



def hyperparameter_counts(dataframe, hyperparameter, x_label, title, name):
    """
    Plots and saves a bar chart of the value counts for a specified hyperparameter in a given DataFrame.

    This function visualizes the distribution of values for a selected hyperparameter within a dataset, 
    highlighting the frequency of each unique value. The resulting bar chart is saved to a file.

    Parameters:
        dataframe : pandas.DataFrame
            The DataFrame containing the data from which to count the hyperparameter values.
        hyperparameter : str
            The name of the column in `dataframe` representing the hyperparameter whose distribution is to be plotted.
        x_label : str
            The label for the x-axis of the plot, typically the name of the hyperparameter.     
        title : str
            The title of the plot, describing what the plot shows.    
        name : str
            The filename under which the plot will be saved. The plot is saved in the '../results/' directory.
            The '.png' extension is recommended to be included in the `name` for clarity.

    Examples:
        >>> df = pd.DataFrame({'model_depth': [2, 3, 4, 2, 3, 3, 2]})
        >>> hyperparameter_counts(df, 'model_depth', 'Model Depth', 'Distribution of Model Depths', 'model_depth_distribution.png')
        # This will create and save a bar chart visualizing the frequency of different model depths in the dataset.

    Notes:
        - The plot is saved with a white background to ensure readability when viewed on various devices.
        - Ensure the '../results/' directory exists before calling this function, or adjust the save path accordingly.
        - The function does not return any value. It directly saves the generated plot to a file.
    """
      
    fig, ax = plt.subplots(figsize=(10, 8))
    dataframe[hyperparameter].value_counts().plot(kind='bar', ax=ax)
    plt.grid()
    plt.xlabel(x_label)
    plt.xticks(rotation=0)
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig('../results/'+name, facecolor='white')



def load_individual_skorch_data(agent_id):
    """
    Loads training, validation, and test datasets for a specified agent or for global model training.

    This function reads the datasets from CSV files. If `agent_id` is -1, it loads the global datasets.
    Otherwise, it loads the agent-specific datasets based on the provided `agent_id`.

    Parameters:
        agent_id : int
            The identifier for the specific agent's dataset to load. If set to -1, the function loads the
            global training, validation, and test datasets.

    Returns:
        tuple
            A tuple containing the training features (X_train_sc), training labels (y_tr), validation features
            (X_val_sc), validation labels (y_vl), test features (X_test_sc), test labels (y_te), column names
            of the training features (X_column_names), and the total exposure from the training set (exposure).

    Examples:
        >>> X_train, y_train, X_val, y_val, X_test, y_test, column_names, exposure = load_individual_skorch_data(-1)
        >>> print(f"Training data shape: {X_train.shape}")
    """
    MY_DATA_PATH = '../data'
    suffix = '' if agent_id == -1 else f'_{agent_id}'
    
    X_train_sc = pd.read_csv(f'{MY_DATA_PATH}/X_train{suffix}.csv')
    X_column_names = X_train_sc.columns.tolist()

    y_tr = pd.read_csv(f'{MY_DATA_PATH}/y_tr{suffix}.csv')
    X_val_sc = pd.read_csv(f'{MY_DATA_PATH}/X_val{suffix}.csv')
    y_vl = pd.read_csv(f'{MY_DATA_PATH}/y_vl{suffix}.csv')
    
    # Note: It assumes X_test and y_test are the same across all agents
    X_test_sc = pd.read_csv(f'{MY_DATA_PATH}/X_test.csv')
    y_te = pd.read_csv(f'{MY_DATA_PATH}/y_test.csv')

    exposure = sum(X_train_sc['Exposure'])
    
    return (X_train_sc, y_tr, X_val_sc, y_vl, X_test_sc, y_te, X_column_names, exposure)


#-------------------- FL training utils: ---------------------------
def seed_torch(seed=SEED):
    """
    Seeds the random number generators of PyTorch, NumPy, and Python's `random` module to ensure
    reproducibility of results across runs when using PyTorch for deep learning experiments.

    This function sets the seed for PyTorch (both CPU and CUDA), NumPy, and the Python `random` module,
    enabling CuDNN benchmarking and deterministic algorithms. It is crucial for experiments requiring
    reproducibility, like model performance comparisons. Note that enabling CuDNN benchmarking and
    deterministic operations may impact performance and limit certain optimizations.

    Parameters:
        seed : int, optional
            The seed value to use for all random number generators. The default value is `SEED`, which
            should be defined beforehand.

    Returns:
        None
            This function does not return a value but sets the random seed for various libraries.

    Notes:
        - When using multiple GPUs, `th.cuda.manual_seed_all(seed)` ensures all GPUs are seeded, 
        crucial for reproducibility in multi-GPU setups.

    Example:
        >>> SEED = 42
        >>> seed_torch(SEED)
    """
    th.manual_seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = True


#----------------------- Data preparation utils: --------------------------------
    
def preprocess_dataframe(df):
    """
    Applies preprocessing steps to the dataframe, including shuffling, data type transformations,
    and value capping based on specified criteria.

    Parameters:
        df : DataFrame 
            The pandas DataFrame to preprocess.

    Returns:
        DataFrame
            The preprocessed DataFrame.

    Usage:
    ```
    df_preprocessed = preprocess_dataframe(df)
    ```
    """    
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Data type transformations and capping
    df['VehPower'] = df['VehPower'].astype(object)  # Convert to categorical
    df['ClaimNb'] = pd.to_numeric(df['ClaimNb'], errors='coerce').clip(upper=4)
    df['VehAge'] = df['VehAge'].clip(upper=20)
    df['DrivAge'] = df['DrivAge'].clip(upper=90)
    df['BonusMalus'] = df['BonusMalus'].clip(upper=150)
    df['Density'] = np.log(df['Density'])
    df['Exposure'] = df['Exposure'].clip(upper=1)
    
    # Drop unused variable
    df = df.drop(['IDpol'], axis=1)
    
    return df

def encode_and_scale_dataframe(df):
    """
    Encodes categorical variables and scales numerical features within the DataFrame.

    Parameters:
        df : DataFrame 
            The DataFrame to encode and scale.

    Returns:
        DataFrame 
            The encoded and scaled DataFrame.
        MinMaxScaler 
            The scaler used for numerical feature scaling.

    Usage:
    ```
    df_encoded, scaler = encode_and_scale_dataframe(df_preprocessed)
    ```
    """
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['VehBrand', 'Region'], drop_first=True)
    
    # Label encoding
    cleanup_nums = {"Area": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6},
                    "VehGas": {"Regular": 1, "Diesel": 2}}
    df_encoded.replace(cleanup_nums, inplace=True)
    
    # Scale features
    scaler = MinMaxScaler()
    features_to_scale = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])
    
    return df_encoded, scaler

def split_data(df_encoded):
    """
    Splits the encoded DataFrame into training, validation, and test sets.

    Parameters:
        df_encoded : DataFrame 
            The encoded DataFrame from which to split the data.

    Returns:
        tuple
            Contains training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).

    Usage:
    ```
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)
    ```
    """
    X = df_encoded.iloc[:, 1:].to_numpy()
    y = df_encoded.iloc[:, 0].to_numpy()
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=SEED)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def upload_dataset():
    """
    Uploads, preprocesses, encodes, scales, and splits the dataset into training, validation, and test sets.

    Assumes the existence of a global `DATA_PATH` variable pointing to the dataset's location and a `SEED` for reproducibility.

    Returns:
        tuple
            Contains the training, validation, and test sets, feature names, and the scaler.

    Usage:
    ```
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = upload_dataset()
    ```
    """
    seed_torch()
    
    df = pd.read_csv(DATA_PATH)
    df_preprocessed = preprocess_dataframe(df)
    df_encoded, scaler = encode_and_scale_dataframe(df_preprocessed)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)
    
    feature_names = df_encoded.columns.tolist()[1:]
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler)


def load_individual_data(agent_id, include_val_in_train=False):
    """
    Loads individual or global datasets as PyTorch TensorDatasets, with an option to include validation data in the training set.

    This function dynamically loads training, validation, and test data from CSV files located in a specified directory.
    It can load data for a specific agent by ID or global data if the agent ID is set to -1. There is an option to merge
    training and validation datasets for scenarios where validation data should be included in training, e.g., for certain
    types of model tuning.

    Parameters:
        agent_id : int
            The identifier for the agent's dataset to load. If set to -1, global datasets are loaded.
        include_val_in_train : bool, optional
            Determines whether validation data is included in the training dataset. Default is False.

    Returns:
        tuple
            A tuple containing the training dataset, validation dataset, test dataset, column names of the training features,
            a tensor of test features, and the total exposure calculated from the training (and optionally validation) dataset.

    Examples:
        >>> train_dataset, val_dataset, test_dataset, column_names, test_features, exposure = load_individual_data(-1, True)
        >>> print(f"Training dataset size: {len(train_dataset)}")
    """
    MY_DATA_PATH = '../data'
    suffix = '' if agent_id == -1 else f'_{agent_id}'
    
    # Load datasets
    X_train = pd.read_csv(f'{MY_DATA_PATH}/X_train{suffix}.csv')
    y_train = pd.read_csv(f'{MY_DATA_PATH}/y_tr{suffix}.csv')
    X_val = pd.read_csv(f'{MY_DATA_PATH}/X_val{suffix}.csv')
    y_val = pd.read_csv(f'{MY_DATA_PATH}/y_vl{suffix}.csv')
    X_test = pd.read_csv(f'{MY_DATA_PATH}/X_test.csv')  # Assuming test data is the same for all agents
    y_test = pd.read_csv(f'{MY_DATA_PATH}/y_test.csv')

    # Merge training and validation datasets if specified
    if include_val_in_train:
        X_train = pd.concat([X_train, X_val], ignore_index=True)
        y_train = pd.concat([y_train, y_val], ignore_index=True)

    # Calculate exposure
    exposure = X_train['Exposure'].sum()

    # Convert to TensorDatasets
    train_dataset = TensorDataset(th.tensor(X_train.values).float(), th.tensor(y_train.values).float())
    val_dataset = TensorDataset(th.tensor(X_val.values).float(), th.tensor(y_val.values).float())
    test_dataset = TensorDataset(th.tensor(X_test.values).float(), th.tensor(y_test.values).float())

    return (train_dataset, val_dataset, test_dataset, X_train.columns.tolist(), th.tensor(X_test.values).float(), exposure)



def uniform_partitions(agents: int = 10, num_features: int = None):
    """
    Splits and saves the dataset into uniform partitions for a specified number of agents.

    This function loads a dataset via a previously defined `upload_dataset` function, then partitions
    the training and validation datasets uniformly across the specified number of agents. Each partition
    is saved to CSV files, containing both features and labels for each agent's training and validation datasets.

    Parameters:
        agents : int, optional
            The number of agents to split the dataset into. Defaults to 10.
        num_features : int, optional
            The number of features in the dataset. Automatically inferred if not specified.

    Notes:
        - Requires `upload_dataset` and `seed_torch` to be defined and accessible within the scope.
        - Saves partitioned data files in the '../data/' directory.

    Example:
        >>> uniform_partitions(agents=5)
        Creates and saves 5 sets of training and validation data for 5 agents, storing them in '../data/'.

    Raises:
        FileNotFoundError
            If the '../data/' directory does not exist or cannot be accessed.
    
    Returns:
        None
            The function does not return a value but saves partitioned datasets to disk.
    """
    # Load the dataset
    X_train_sc, X_val_sc, X_test_sc, y_tr, y_vl, y_te, X_column_names, _ = upload_dataset()
    num_features = num_features or X_train_sc.shape[1]
    
    # Define the base path for saving files
    base_path = '../data/'
    
    # Function to save datasets to CSV
    def save_to_csv(data, filename, column_names=X_column_names):
        pd.DataFrame(data, columns=column_names).to_csv(f'{base_path}{filename}', index=False)
    
    # Save the global datasets
    save_to_csv(X_train_sc, 'X_train.csv')
    save_to_csv(y_tr, 'y_tr.csv', ['y'])
    save_to_csv(X_val_sc, 'X_val.csv')
    save_to_csv(y_vl, 'y_vl.csv', ['y'])
    save_to_csv(X_test_sc[:, :num_features], 'X_test.csv')
    save_to_csv(y_te, 'y_test.csv', ['y'])

    # Prepare and shuffle data
    seed_torch()
    train_data = np.hstack((X_train_sc, y_tr.reshape(-1, 1)))
    val_data = np.hstack((X_val_sc, y_vl.reshape(-1, 1)))
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    # Split and save partitioned data
    for i in range(agents):
        partition_train = np.array_split(train_data, agents)[i]
        partition_val = np.array_split(val_data, agents)[i]

        save_to_csv(partition_train[:, :num_features], f'X_train_{i}.csv')
        save_to_csv(partition_train[:, num_features:], f'y_tr_{i}.csv', ['y'])
        save_to_csv(partition_val[:, :num_features], f'X_val_{i}.csv')
        save_to_csv(partition_val[:, num_features:], f'y_vl_{i}.csv', ['y'])


#-------------------- results analysis utils: -------------------------------------------
def lorenz_curve(y_true, y_pred, exposure):
    """
    Calculates the Lorenz curve for given true values, predicted values, and exposures.

    The Lorenz curve is a graphical representation of the distribution of income or wealth. In this context,
    it is used to show the distribution of claims or losses in insurance, ordered by predicted risk. This function
    calculates the cumulative percentage of claims and exposures, sorted by the predicted risk.

    Parameters:
        y_true : array_like
            The true values of the claims or losses.
        y_pred : array_like
            The predicted risk scores associated with each claim or loss.
        exposure : array_like
            The exposure values associated with each observation.

    Returns:
        tuple of numpy.ndarray
            A tuple containing two arrays: the cumulative percentage of exposure and the cumulative percentage of claims,
            both sorted by the predicted risk.

    Examples:
        >>> y_true = np.array([100, 50, 20])
        >>> y_pred = np.array([0.2, 0.5, 0.1])
        >>> exposure = np.array([1, 2, 1])
        >>> cumulated_exposure, cumulated_claims = lorenz_curve(y_true, y_pred, exposure)
        >>> print(cumulated_exposure)
        >>> print(cumulated_claims)
    """
    # Convert inputs to numpy arrays and reshape for consistency
    y_true, y_pred, exposure = map(lambda x: np.asarray(x).flatten(), (y_true, y_pred, exposure))

    # Order samples by increasing predicted risk
    ranking = np.argsort(y_pred)
    ranked_frequencies = y_true[ranking]
    ranked_exposure = exposure[ranking]

    # Calculate cumulative claims and exposure, then normalize
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure) / np.sum(ranked_frequencies * ranked_exposure)
    cumulated_exposure = np.cumsum(ranked_exposure) / np.sum(ranked_exposure)
    
    return cumulated_exposure, cumulated_claims

#---------------------------- GRAPHS AND RESULTS ANALYSIS UTILS  ------------------------------------------------------

def load_model(agent=-1, num_features=NUM_FEATURES):
    """
        Load a pre-trained neural network model for a specific agent.

        Parameters:
            agent : int 
                The ID of the agent whose model to load. Default is -1.
            num_features : int 
                The number of input features for the model. Default is NUM_FEATURES.

        Returns:
            loaded_agent_model : NeuralNetRegressor 
                The loaded neural network model for the specified agent.
    """
    
    all_results_df = pd.read_csv('../results/all_results.csv')
    top_results_df = all_results_df.loc[all_results_df['rank_test_score']==1]
    top_results_dict = top_results_df[['agent', 'param_module__num_units_1', 'param_module__num_units_2']].set_index('agent').to_dict('index')

    num_units_1 = list(top_results_dict[agent].items())[0][1]
    num_units_2 = list(top_results_dict[agent].items())[1][1]

    loaded_agent_model = NeuralNetRegressor(architecture.MultipleRegression(num_features, num_units_1, num_units_2).double())
    loaded_agent_model.initialize()  # This is important!
    loaded_agent_model.load_params(f_params=f'../ag_'+str(agent)+'/agent_'+str(agent)+'_model.pkl', 
                                       f_optimizer=f'../ag_'+str(agent)+'/agent_'+str(agent)+'_opt.pkl', 
                                       f_history=f'../ag_'+str(agent)+'/agent_'+str(agent)+'_history.json')
    
    return loaded_agent_model

def frequency_conversion(FACTOR, df, freq_dictionary):
    """
    Perform frequency conversion on a DataFrame.

    Parameters:
        FACTOR : str 
            The factor to be converted.
        df : pandas.DataFrame 
            The DataFrame containing the data.
        freq_dictionary : dict 
            A dictionary mapping factor keys to frequency keys.

    Returns:
        df : pandas.DataFrame 
            The DataFrame with frequency conversion applied.
    """

    for key in freq_dictionary:
        df[freq_dictionary[key]]=df[key]/df['Exposure']

    df.insert(1,FACTOR+'_binned_midpoint',[round((a.left + a.right)/2,0) for a in df[FACTOR+'_binned']])

def undummify(df, prefix_sep="_"):
    """
    Reverse one-hot encoding (dummy variables) in a DataFrame.

    Parameters:
        - df (pandas.DataFrame): The DataFrame containing dummy variables.
        - prefix_sep (str, optional): Separator used in column prefixes. Default is "_".

    Returns:
        undummified_df (pandas.DataFrame): The DataFrame with one-hot encoding reversed.
    """
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def create_test_data(): 
    """
    Create test data for evaluation.

    This function loads test data, undummifies categorical variables,
    applies scaling to certain features, bins numerical factors, and returns
    processed test datasets for evaluation.

    Returns:
        X_test : pandas.DataFrame
            Test features dataset.
        y_test : pandas.DataFrame
            Test labels dataset.
        df_test : pandas.DataFrame
            Processed test dataset.
    """

    (X_train, X_val, X_test, y_train, y_val, y_test, X_column_names, scaler) = upload_dataset()
    
    MY_DATA_PATH = '../data'
    
    X_test_sc = pd.read_csv(MY_DATA_PATH + '/X_test.csv')
    y_te = pd.read_csv(MY_DATA_PATH + '/y_test.csv')
    X_column_names = X_test_sc.columns.tolist()

    test_complete_data=np.column_stack((X_test_sc, y_te))

    X_column_names.append('ClaimNb')

    #Convert dataset of test data, actuals, and prediction back into dataframe

    df_test=pd.DataFrame(data=test_complete_data,    # values
                    columns=X_column_names)  # 1st row as the column names
    
    # Un one-hot encode Region and VehBrand
    df_test = undummify(df_test)
    df_test['VehBrand_number'] = df_test['VehBrand'].str[1:].astype(int)
    df_test['Region_number'] = df_test['Region'].str[1:].astype(int)


    df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']]=scaler.inverse_transform(df_test[['Area', 'VehPower', 'VehAge','DrivAge','BonusMalus','Density']] )
    
    #Bin factors
    factor_list = ['Area', 'VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'VehGas', 'Density', 'VehBrand_number', 'Region_number']
    
    # If fewer levels than MAX_BINSIZE no effective binning happens via the min() function, not used currently
    MAX_BINSIZE = 20

    # Automatic binning not used, could use qcut as well 
    for i in factor_list:
        df_test[i+'_binned'] = pd.cut(df_test[i], bins=min(len(df_test[i].unique()), MAX_BINSIZE), duplicates='drop')
        #df_test[i+'_binned'] = pd.cut(df_test[i], bins=np.linspace(0, max(df_test[i]), min(21, len(df_test[i].unique()))), duplicates='drop')
    
    # Custom binning used instead to define more sensible bins
    df_test['Area'+'_binned'] = pd.cut(df_test['Area'], bins=np.linspace(0, 7, 8), duplicates='drop')
    df_test['VehPower'+'_binned'] = pd.cut(df_test['VehPower'], bins=np.linspace(0, 15, 16), duplicates='drop')
    df_test['VehAge'+'_binned'] = pd.cut(df_test['VehAge'], bins=np.linspace(0, 20, 21), duplicates='drop')
    df_test['DrivAge'+'_binned'] = pd.cut(df_test['DrivAge'], bins=np.linspace(0, 100, 21), duplicates='drop')
    df_test['BonusMalus'+'_binned'] = pd.qcut(df_test['BonusMalus'], q=15, duplicates='drop') # note usin qcut here
    df_test['Density'+'_binned'] = pd.cut(df_test['Density'], bins=np.linspace(0, 11, 12), duplicates='drop')
    df_test['VehBrand_number'+'_binned'] = pd.cut(df_test['VehBrand_number'], bins=np.linspace(0, 15, 16), duplicates='drop')
    df_test['Region_number'+'_binned'] = pd.cut(df_test['Region_number'], bins=np.linspace(0, 100, 21), duplicates='drop')
    df_test['VehGas'+'_binned'] = pd.cut(df_test['VehGas'], bins=np.linspace(0, 2, 3), duplicates='drop')
    
    return X_test, y_test, df_test

def create_df_test_pred(df_test, X_test, NUM_AGENTS, global_model, fl_model, agent_model_dictionary):
    """
    Generate predictions for the test dataset using various models.

    Parameters:
        df_test : pandas.DataFrame
            The test dataset.
        X_test : numpy.ndarray
            The features of the test dataset.
        NUM_AGENTS : int
            The number of agents.
        global_model 
            The global model for prediction.
        fl_model 
            The federated learning model for prediction.
        agent_model_dictionary : dict
            A dictionary containing agent models.

    Returns:
        df_test : pandas.DataFrame 
            The test dataset with predictions appended.
    """

    # Global Model Predictions
    y_pred = global_model.predict(X_test.astype(np.float64))
    df_test['ClaimNb_pred']=pd.Series(y_pred.flatten())

    # FL Model Predictions
    y_fl_pred = fl_model.predict(th.tensor(X_test).float())
    df_test['ClaimNb_fl_pred']=pd.Series(y_fl_pred.flatten().detach().numpy())

    # Local Model Predictions

    agent_prediction_dictionary = {}

    for agents in range(NUM_AGENTS):
        agent_prediction_dictionary["y_agent_{0}_pred".format(agents)] = agent_model_dictionary['loaded_agent_'+str(agents)+'_model'].predict(X_test.astype(np.float64))
        df_test['ClaimNb_agent_'+str(agents)+'_pred']=pd.Series(agent_prediction_dictionary['y_agent_'+str(agents)+'_pred'].flatten())

    return df_test

def create_df_sum(df_test_pred, factor, NUM_AGENTS):
    """
    Create a summary DataFrame aggregating predictions by binned factors.

    Parameters:
        df_test_pred : pandas.DataFrame
            The DataFrame with test predictions.
        factor : str 
            The factor for binning.
        NUM_AGENTS : int 
            The number of agents.

    Returns:
        df_sum : pandas.DataFrame 
            The summary DataFrame aggregated by binned factors.
    """

    sum_list = ['Exposure',  'ClaimNb', 'ClaimNb_pred', 'ClaimNb_fl_pred']
    sum_dictionary = {'ClaimNb':'Actual freq', 'ClaimNb_pred':'Freq pred global model', 'ClaimNb_fl_pred':'Freq pred FL model'}

    for agents in range(NUM_AGENTS):
        sum_list.append('ClaimNb_agent_'+str(agents)+'_pred')
        sum_dictionary['ClaimNb_agent_'+str(agents)+'_pred']='Freq pred agent '+str(agents)+' model'

    df_sum=df_test_pred.groupby([factor+'_binned'])[sum_list].sum().reset_index()

    frequency_conversion(factor, df_sum, sum_dictionary)

    # Remove rows with 0 exposure
    df_sum = df_sum.loc[df_sum['Exposure']!=0]
    df_sum = df_sum.reset_index(drop=True)

    return df_sum

def one_way_graph_comparison(factor, df_test_pred, agents_to_graph_list, NUM_AGENTS):
        """
        Generate a one-way graph comparison of actual vs. predicted frequencies by agents.

        Parameters:
            factor : str
                The factor for binning.
            df_test_pred : pandas.DataFrame
                The DataFrame with test predictions.
            agents_to_graph_list : list
                List of agent indices to include in the graph.
            NUM_AGENTS : int 
                The total number of agents.

        Returns:
        None
        """

        df_sum = create_df_sum(df_test_pred, factor, NUM_AGENTS)
        
        fig, ax = plt.subplots(figsize=(12, 8))

        plt.plot(df_sum['Actual freq'],  
                label='Actual freq',
                marker='s',
                markersize=10,
                )
        
        for agents in agents_to_graph_list:
                plt.plot(df_sum['Freq pred agent '+str(agents)+' model'],  
                label='Freq pred agent '+str(agents)+' model',
                marker='o',
                markersize=5,
                #linestyle=(0, (1, 10)),
                linestyle='dotted',
                )


        plt.plot(df_sum['Freq pred FL model'],  
                label='Freq pred FL model',
                marker='o',
                markersize=10,
                #linestyle='dotted',
                )
        
        plt.plot(df_sum['Freq pred global model'],  
                label='Freq pred global model',
                marker='s',
                markersize=10,
                #linestyle='dotted',
                )

        plt.legend(bbox_to_anchor=(1.08, 1), loc='upper left', borderaxespad=0)

        plt.xlabel(factor+' binned')
        plt.xticks(rotation = 75)
        plt.ylabel('Frequency')
        plt.title('Actual vs. Expected by '+factor)
        plt.grid()


        # Get second axis
        ax2 = ax.twinx()

        plt.bar(df_sum[factor+'_binned'].astype(str), 
                df_sum['Exposure'], 
                label='Exposure', 
                color='y',
                alpha=0.35
                )

        plt.ylabel('Exposure', color='y')
        plt.xticks(rotation = 90)

        vals = ax2.get_yticks()
        ax2.set_yticklabels(['{:,.0f}'.format(x) for x in vals])

        plt.legend(bbox_to_anchor=(1.08, 0), loc='upper left', borderaxespad=0)

        plt.show()

def double_lift_rebase(df_test_pred, model1, model2):
    """
    Generate a double lift chart comparing the performance of two models.

    Parameters:
        df_test_pred : pandas.DataFrame
            The DataFrame with test predictions.
        model1 : str
            The name of the first model.
        model2 : str 
            The name of the second model.

    Returns:
    - None
    """

    # Rename 
    df_test_pred['ClaimNb_global_pred'] = df_test_pred['ClaimNb_pred']
    # Compare models
    df_test_pred[model1+'_vs_'+model2] = df_test_pred['ClaimNb_'+model1+'_pred']/df_test_pred['ClaimNb_'+model2+'_pred']
    # Bin comparison
    df_test_pred[model1+'_vs_'+model2+'_binned'] = pd.qcut(df_test_pred[model1+'_vs_'+model2], 10)
    # Sum up predictions
    sum_list = ['Exposure',  
                'ClaimNb', 
                'ClaimNb_pred', 
                'ClaimNb_'+model1+'_pred',
                'ClaimNb_'+model2+'_pred',
                ]
    
    df_sum=df_test_pred.groupby([model1+'_vs_'+model2+'_binned'])[sum_list].sum().reset_index()
    # Rebase
    df_sum['ClaimNb_rebased'] = df_sum['ClaimNb']/df_sum['ClaimNb']
    df_sum['ClaimNb_'+model1+'_pred'+'_rebased'] = df_sum['ClaimNb_'+model1+'_pred']/df_sum['ClaimNb']
    df_sum['ClaimNb_'+model2+'_pred'+'_rebased'] = df_sum['ClaimNb_'+model2+'_pred']/df_sum['ClaimNb']
    # Graph
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.plot(df_sum['ClaimNb_rebased'],  
            label='Actual freq',
            marker='s',
            markersize=10,
            )

    plt.plot(df_sum['ClaimNb_'+model2+'_pred'+'_rebased'],  
            label='Freq pred '+model2+' model',
            marker='o',
            markersize=10,
            #linestyle=(0, (1, 10)),
            #linestyle='dotted',
            )

    plt.plot(df_sum['ClaimNb_'+model1+'_pred'+'_rebased'],  
            label='Freq pred '+model1+' model',
            marker='o',
            markersize=10,
            #linestyle='dotted',
            )

    plt.legend(bbox_to_anchor=(1.08, 1), loc='upper left', borderaxespad=0)

    plt.xlabel('Model 1 Prediction Over Model 2 Prediction')
    plt.xticks(rotation = 75)
    plt.ylabel('Actual Over Expected')
    plt.title('Double Lift Chart Comparing Model Performance')
    plt.grid()


    # Get second axis
    ax2 = ax.twinx()

    plt.bar(df_sum[model1+'_vs_'+model2+'_binned'].astype(str), 
            df_sum['Exposure'], 
            label='Exposure', 
            color='y',
            alpha=0.35
            )

    plt.ylabel('Exposure', color='y')
    plt.xticks(rotation = 90)

    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0f}'.format(x) for x in vals])

    plt.legend(bbox_to_anchor=(1.08, 0), loc='upper left', borderaxespad=0)

    plt.show()

