import argparse
import utils
import run_config

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for federated learning dataset preparation and validation.

    Returns:
        argparse.Namespace 
        An object containing the parsed command line arguments with the following attributes
            num_agents : int 
                Specifies the number of agents among which the dataset will be partitioned.
    """
    parser = argparse.ArgumentParser(description="Prepare and validate dataset for federated learning.")
    parser.add_argument(
        "--num_agents",
        type=int,
        default=int(run_config.dataset_config.get("num_agents", 10)),
        help="Specifies the number of agents to distribute the dataset across. Defaults to the value in run_config."
    )
    return parser.parse_args()

def prepare_and_validate_dataset(num_agents: int):
    """
    Prepares the dataset for federated learning by partitioning it uniformly (by number of records) among the specified number of agents.
    Performs validation checks to ensure the integrity of the dataset partitioning.

    Parameters:
        num_agents : int
            The number of agents among which the dataset will be partitioned.
    """
    # Prepare the dataset by partitioning it uniformly among the specified number of agents
    utils.uniform_partitions(num_agents)
    
    # Perform a row check to validate the integrity of the dataset partitioning
    utils.row_check(num_agents)

def main():
    """
    Main execution function that orchestrates the dataset preparation and validation for federated learning.
    """
    # Parse command line arguments to determine the configuration for federated learning
    args = parse_arguments()
    
    # Prepare the dataset and validate it based on the specified number of agents
    prepare_and_validate_dataset(args.num_agents)

    print("Dataset preparation and validation completed successfully.")

if __name__ == "__main__":
    main()
