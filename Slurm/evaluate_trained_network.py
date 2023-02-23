#! /usr/bin/env python
import argparse
import torch
import os
from GNN.GNN import GNNPolicy
from Slurm.train_neural_network import create_tensorboard_writer, get_standard_solve_data, \
    get_rand_seeds_from_feature_generators, generate_batches, run_test_set
from utilities import remove_temp_files, str_to_bool, is_file, is_dir, get_instances, remove_slurm_files


def evaluate_neural_network(instance_dir, solution_dir, feature_dir, default_results_dir, results_dir, run_dir,
                            temp_dir, neural_network_path, outfile_dir, root):
    """
    Args:
        instance_dir: The directory containing all instance data
        solution_dir: The directory containing all instance solution data
        feature_dir: The directory containing all feature representation data
        default_results_dir: The directory containing all results data from default SCIP runs for each instance
        results_dir: The directory where we will dump our results from this run
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
    standard_solve_data = get_standard_solve_data(default_results_dir, root=root)

    # Extract the random seeds we will use on our SCIP instances. These should be in our standard_solve_data
    rand_seeds = get_rand_seeds_from_feature_generators(standard_solve_data)

    # Initialise the neural network, and load a saved network if one is given
    neural_network = GNNPolicy()
    neural_network.load_state_dict(torch.load(neural_network_path))
    neural_network.eval()

    # Get the instance paths and names from the data directory
    instances = get_instances(instance_dir)

    # Grab the batches that will be used. For testing this should be one large batch
    batch_instances, random_state = generate_batches(instances, 1, None)
    assert len(batch_instances) == 1

    # Empty the temporary directory containing all batch-specific files
    remove_temp_files(temp_dir)

    # Run the test-set TODO: Define a formal test-set instead of using the batch w/o perturbation
    time_limit = -1 if root else 7200
    _, _ = run_test_set(instance_dir, solution_dir, feature_dir, results_dir, temp_dir,
                        outfile_dir, neural_network,
                        batch_instances[0], rand_seeds, standard_solve_data, 0,
                        tensorboard_writer, time_limit=time_limit, root=root,
                        rm_temp_files=False, exclusive=True, create_yaml=True, single_instance=None)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=is_dir)
    parser.add_argument('feature_dir', type=is_dir)
    parser.add_argument('default_results_dir', type=is_dir)
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('run_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('neural_network_path', type=is_file)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('root', type=str_to_bool)
    args = parser.parse_args()

    args.outfile_dir = os.path.join(args.outfile_dir, 'eval_full_network')
    if not os.path.isdir(args.outfile_dir):
        os.mkdir(args.outfile_dir)
    else:
        remove_slurm_files(args.outfile_dir)

    # The main function call to evaluate a trained neural network
    evaluate_neural_network(args.instance_dir, args.solution_dir, args.feature_dir, args.default_results_dir,
                            args.results_dir, args.run_dir, args.temp_dir, args.neural_network_path, args.outfile_dir,
                            args.root)
