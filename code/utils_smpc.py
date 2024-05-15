import csv
import numpy as np
import run_config
from utils_quantisation import modulus, M


def calc_noise(file_path: str, agent_number: int) -> dict:
    """
    Calculate noise based on seeds stored in a CSV file for a specific agent.

    This function reads a CSV file containing seeds for each collaborator by round,
    generates noise vectors for each round, and applies quantization if enabled in 
    the run configuration. It differentiates the noise for the specific agent from others.

    Parameters:
        file_path : str
            The path to the CSV file containing the seeds.
        agent_number : int
            The index of the current agent for whom the noise is being calculated.

    Returns:
        dict
            A dictionary with round numbers as keys and noise vectors as values.

    Example:
        >>> noises = calc_noise('path/to/seeds.csv', 2)
        >>> print(noises[1])  # Noise vector for round 1
    """
    with open(file_path, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.reader(infile)
        seeds = {int(rows[0]): rows[1:] for rows in reader} 

    noises = {}
    for round_number, seeds_per_round in seeds.items():
        noise_vector = np.zeros(162, dtype=np.int64)
        for i, seed in enumerate(seeds_per_round):
            np.random.seed(int(seed))
            noise_adjustment = np.random.randint(M + 1, size=162, dtype=np.int64)
            
            if run_config.QUANTISATION:
                noise_vector = modulus(noise_vector + ((run_config.server_config["num_clients"] - 1) * noise_adjustment if i == agent_number else -noise_adjustment))
            else:
                uniform_noise = np.round(np.random.uniform(-2, 2, 162), 5)
                noise_vector = noise_vector.astype(float)
                noise_vector += uniform_noise * float(run_config.server_config["num_clients"] - 1 if i == agent_number else -1)

        noises[round_number] = noise_vector
    
    return noises



def calc_noise_zero(file_path: str, agent_number: int) -> dict:
    """
    Generate a dictionary of zero noise vectors for each round based on the CSV file.

    Parameters:
        file_path : str
            The path to the CSV file containing the seeds.
        agent_number : int
            The agent number, unused in this function but maintained for API consistency.

    Returns:
        dict
            A dictionary with round numbers as keys and zero noise vectors as values.

    Example:
        >>> zero_noises = calc_noise_zero('path/to/seeds.csv', 2)
        >>> print(zero_noises[1])  # Zero noise vector for round 1
    """
    with open(file_path, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.reader(infile)
        seeds = {int(rows[0]): rows[1:] for rows in reader} 
    
    noises = {round_number: np.zeros(162) for round_number in seeds.keys()}
    
    return noises
