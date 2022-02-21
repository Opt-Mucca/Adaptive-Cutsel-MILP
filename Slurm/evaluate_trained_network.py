#! /usr/bin/env python
import argparse
import torch
from GNN.GNN import GNNPolicy
from Slurm.train_neural_network import create_tensorboard_writer, get_standard_solve_data, \
    get_rand_seeds_from_feature_generators, get_instances, generate_batches, run_test_set, remove_slurm_files
from utilities import remove_temp_files, str_to_bool, is_file, is_dir


def evaluate_neural_network(data_dir, run_dir, temp_dir, neural_network_path, outfile_dir, root):
    """
    Args:
        data_dir: The directory containing all of our reference data
        run_dir: The directory in which all tensorboard information will be dumped into
        temp_dir: The directory where we will dump all of our temporary files
        neural_network_path: The path to the neural network state dictionary
        outfile_dir: The directory where we will dump all the slurm outfiles
        root: A boolean that indicates whether we would like our evaluation to be restricted to the root node or not
    Returns:
        Nothing. It adds a complete run data to our tensorboard log showing the performance of our neural network
    """

    # Create the tensorboard writer to store training data
    tensorboard_writer = create_tensorboard_writer(-1, run_dir, instance=None)

    # Load the data associated with our instances that was performed when generating features
    standard_solve_data = get_standard_solve_data(data_dir, root=root)

    # Extract the random seeds we will use on our SCIP instances. These should be in our standard_solve_data
    rand_seeds = get_rand_seeds_from_feature_generators(standard_solve_data)

    # Initialise the neural network, and load a saved network if one is given
    neural_network = GNNPolicy()
    neural_network.load_state_dict(torch.load(neural_network_path))
    neural_network.eval()

    # Get the instance paths and names from the data directory
    instances = get_instances(data_dir)

    # Grab the batches that will be used. For testing this should be one large batch
    batch_instances, random_state = generate_batches(instances, 1, None)
    assert len(batch_instances) == 1

    # Empty the temporary directory containing all batch-specific files
    remove_temp_files(temp_dir)

    # Run the test-set TODO: Define a formal test-set instead of using the batch w/o perturbation
    time_limit = -1 if root else 600
    test_i, tensorboard_writer = run_test_set(data_dir, temp_dir, outfile_dir, neural_network,
                                              batch_instances[0], rand_seeds, standard_solve_data, 0,
                                              4, 5, tensorboard_writer, time_limit=time_limit, root=root,
                                              rm_temp_files=False, exclusive=False, create_yaml=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=is_dir)
    parser.add_argument('run_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('neural_network_path', type=is_file)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('root', type=str_to_bool)
    args = parser.parse_args()

    # The main function call to evaluate a trained neural network
    evaluate_neural_network(args.data_dir, args.run_dir, args.temp_dir, args.neural_network_path, args.outfile_dir,
                            args.root)
