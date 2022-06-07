#! /usr/bin/env python
import os
import argparse
import yaml
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from utilities import read_feature_vector_files, str_to_bool
from utilities import remove_slurm_files, remove_temp_files, run_python_slurm_job, get_covariance_matrix
from utilities import get_filename, is_dir
from GNN.GNN import GNNPolicy


def train_network(data_dir, run_dir, temp_dir, prev_network, outfile_dir, num_epochs, rel_batch_size, num_samples,
                  seed_init, single_instance=None):
    """
    The main training function for our GNN.

    Args:
        data_dir: The directory containing all instances and data files for feature vectors and graph representations
        run_dir: The directory in which all tensorboard information will be dumped into
        temp_dir: The directory in which all temporary files per batch will be dumped then deleted (e.g. cut-params)
        prev_network: An optional path to a previously trained network which can be loaded
        outfile_dir: The directory in which the slurm outfiles for individual jobs will be dumped
        num_epochs: The number of epochs in training. One epoch goes through each instance exactly once.
        rel_batch_size: The batch size relative to the total number of instances
        num_samples: The number of samples used for each combination of instance and seed.
        seed_init: A random seed so the same batch ordering and NN initialisation can be reproduced
        single_instance: This is a kwarg that indicates if we want to train for a single instance

    Returns:
        Nothing, but it should end up saving the run s.t the results are accessibly through tensorboard
        and should save the trained network.

    """

    # Create the tensorboard writer to store training data
    tensorboard_writer = create_tensorboard_writer(seed_init, run_dir, instance=single_instance)

    # Load the data associated with our instances that was performed when generating features
    standard_solve_data = get_standard_solve_data(data_dir, root=True)

    # Extract the random seeds we will use on our SCIP instances. These should be in our standard_solve_data
    rand_seeds = get_rand_seeds_from_feature_generators(standard_solve_data)

    # Initialise the numpy random state
    random_state = np.random.RandomState(seed_init) if seed_init >= 0 else None

    # Initialise the torch random seed if needed
    if seed_init >= 0:
        torch.manual_seed(seed_init)

    # Initialise the neural network, and load a saved network if one is given
    neural_network = GNNPolicy()
    if prev_network is not None:
        neural_network.load_state_dict(torch.load(prev_network))

    # Initialise the optimizer we use to wrap our neural_network with
    optimiser = torch.optim.Adam(neural_network.parameters(), lr=0.00005)

    # Get the instance paths and names from the data directory
    instances = get_instances(data_dir)

    # If single_instance is not None, then we only want to train on this instance
    if single_instance is not None:
        assert single_instance in instances
        instances = [single_instance]

    # Keep an index that remembers the batch index in the larger setting of the run w.r.t previous epochs etc
    run_i = 0
    test_i = 0

    # The main training loop
    for epoch_i in range(num_epochs):

        # Grab the batches that will be used this epoch
        batch_instances, random_state = generate_batches(instances, rel_batch_size, random_state)

        # Now cycle over the batches we have produced this epoch
        for batch_i in range(len(batch_instances)):
            # Empty the temporary directory containing all batch-specific files
            remove_temp_files(temp_dir)

            # Run the test-set TODO: Define a formal test-set instead of using the batch w/o perturbation
            if single_instance is not None:
                test_i, tensorboard_writer = run_test_set(data_dir, temp_dir, outfile_dir, neural_network,
                                                          batch_instances[batch_i], rand_seeds, standard_solve_data,
                                                          test_i, run_i, 5, tensorboard_writer,
                                                          create_yaml=single_instance is not None)

            # Create our cut-selector parameter sample values / files as well as the distributions they come from
            neural_network, sampled_cut_params, multivariate_normals = create_cut_selector_params(data_dir, temp_dir,
                                                                                                  neural_network,
                                                                                                  batch_instances[
                                                                                                      batch_i],
                                                                                                  rand_seeds, epoch_i,
                                                                                                  num_epochs,
                                                                                                  num_samples)

            # Create a new sub-directory in the slurm outfile directory to contain logs from this batch
            if single_instance is not None:
                batch_outfile_dir = os.path.join(outfile_dir, str(run_i))
                os.mkdir(batch_outfile_dir)
            else:
                epoch_outfile_dir = os.path.join(outfile_dir, str(epoch_i))
                if not os.path.isdir(epoch_outfile_dir):
                    os.mkdir(epoch_outfile_dir)
                batch_outfile_dir = os.path.join(epoch_outfile_dir, str(run_i))
                os.mkdir(batch_outfile_dir)

            # Create slurm jobs. Each job consists of one instance, one seed, and a sampled cut-sel param set
            signal_file = submit_slurm_jobs(data_dir, temp_dir, batch_outfile_dir, num_samples,
                                            batch_instances[batch_i], rand_seeds, -1, True)

            # Wait for slurm jobs to finish and read in all solve information from the jobs
            batch_data, sampled_cut_params = wait_for_slurm_jobs_and_extract_solve_info(temp_dir,
                                                                                        batch_instances[batch_i],
                                                                                        rand_seeds, num_samples,
                                                                                        sampled_cut_params,
                                                                                        signal_file)

            # Generate scores based on solve_information
            (scores, gaps, dual_bounds, primal_bounds, lp_iterations, n_nodes, n_cuts, sol_times, sol_fracs,
             primal_dual_ints) = calculate_scores(batch_data, standard_solve_data, run_i)

            # Use our generated scores to update our neural network
            optimiser = reinforce_and_update_neural_network(optimiser, scores, sampled_cut_params, multivariate_normals)

            # Add all data related to the batch to the summary writer
            tensorboard_writer = add_data_to_tensorboard_writer(tensorboard_writer, batch_data, scores, gaps,
                                                                dual_bounds, primal_bounds,
                                                                lp_iterations, n_nodes, n_cuts, sol_times,
                                                                sol_fracs, primal_dual_ints, run_i, neural_network)

            # Save the latest iteration of the neural_network in a secure directory
            torch.save(neural_network.state_dict(), os.path.join(data_dir, 'actor.pt'))

            # There's an issue with memory management as subprocess duplicates memory when it needs only a fraction
            # We will flag some variables for the garbage collector preemptively
            del scores, gaps, dual_bounds, primal_bounds, lp_iterations, n_nodes, n_cuts, sol_times, sol_fracs
            del primal_dual_ints

            # Increment the run index
            run_i += 1

        if single_instance is None:
            torch.save(neural_network.state_dict(), os.path.join(data_dir, 'actor_{}.pt'.format(epoch_i)))
            print('Completed Epoch {}'.format(epoch_i), flush=True)

    if single_instance is not None:
        torch.save(neural_network.state_dict(), os.path.join(data_dir, single_instance + '.pt'))
        remove_temp_files(temp_dir)
        _, tensorboard_writer = run_test_set(data_dir, temp_dir, outfile_dir, neural_network, instances, rand_seeds,
                                             standard_solve_data, test_i, 4, 5, tensorboard_writer,
                                             create_yaml=single_instance is not None)

    return


def create_cut_selector_params(data_dir, temp_dir, neural_network, instances, rand_seeds, epoch_i, num_epochs,
                               num_samples_per_instance):
    """
    Args:
        data_dir: The directory containing all our data that is not run dependent
        temp_dir: The directory in which we will place all temporary files from this batch
        neural_network: The neural network objective
        instances: A list of instance strings in our batch
        rand_seeds: A list of the random seeds we will use in our solves
        epoch_i: The epoch of the current batch
        num_epochs: The total number of epochs in this run
        num_samples_per_instance: The number of SCIP runs we want to call per instance / seed combination

    Returns:
        The neural network, The cut-selector parameters, The distributions centred at our generated parameters
    """

    assert type(num_samples_per_instance) == int and num_samples_per_instance > 0

    # Initialise the dictionary containing the samples we take of the cut-selector parameters
    sampled_cut_selector_params = {instance: {rand_seed: [] for rand_seed in rand_seeds} for instance in instances}
    # Initialise the dictionary containing all the multi-normal distributions we generate
    multivariate_normal_distributions = {instance: {rand_seed: None for rand_seed in rand_seeds
                                                    } for instance in instances}

    # Goes through all instances then random seeds. Produces the cut-selector params and saves them to a npy file
    for instance in instances:
        # The instances should be the same over random seeds. We do this in case pre-solve is not seed independent.
        for rand_seed in rand_seeds:
            # Load the features of the instance: the bipartite graph, and row / column / edge features
            edge_indices, coefficients, col_features, row_features = read_feature_vector_files(data_dir, instance,
                                                                                               rand_seed,
                                                                                               torch_output=True)

            # Get the cut-sel params from a forward pass of the network
            cut_selector_params = neural_network.forward(edge_indices, coefficients, col_features, row_features)

            # There's an issue with memory management as subprocess duplicates memory when it needs only a fraction
            # We will flag some variables for the garbage collector preemptively
            del edge_indices, coefficients, col_features, row_features

            # Create a multi-normal distribution centred at our predictions. Assume independence of parameters
            covariance_matrix = get_covariance_matrix(epoch_i, num_epochs)
            m = torch.distributions.multivariate_normal.MultivariateNormal(cut_selector_params, covariance_matrix)
            multivariate_normal_distributions[instance][rand_seed] = m

            # We want to sample the distribution we've created centred at our predictions
            for sample_i in range(num_samples_per_instance):
                # If we are after only one output, then simply take our prediction
                if num_samples_per_instance == 1:
                    sampled_cut_selector_params[instance][rand_seed].append(cut_selector_params)
                    sample = cut_selector_params
                else:
                    sample = m.sample()
                    sampled_cut_selector_params[instance][rand_seed].append(sample)

                # These values cannot be set to negative values in the solver, so simply put them through a ReLu
                # We did not use the non-negative values and allowed negative multipliers.
                # non_negative_sampled_values = torch.nn.functional.relu(sample)
                # We then want to create a .npy file that stores these cut-selector values
                file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=sample_i,
                                         ext='npy')
                np.save(file_name, sample.detach().numpy())

    return neural_network, sampled_cut_selector_params, multivariate_normal_distributions


def reinforce_and_update_neural_network(optimiser, scores, sampled_cut_selector_params,
                                        multivariate_normal_distributions):
    # Remove all previous gradient information
    optimiser.zero_grad()

    # Now start the main loop to populate the rewards and log probabilities with our run data
    log_probabilities = []
    rewards = []
    for instance in scores:
        for rand_seed in scores[instance]:
            for sample_i in range(len(scores[instance][rand_seed])):
                log_probabilities.append(multivariate_normal_distributions[instance][rand_seed].log_prob(
                    sampled_cut_selector_params[instance][rand_seed][sample_i]))
                rewards.append(scores[instance][rand_seed][sample_i])

    log_probabilities = torch.stack(log_probabilities)
    rewards = torch.tensor(rewards)

    # Get the loss as an aggregate over our runs
    loss = (-1 * log_probabilities * rewards).mean()
    # Get the gradient information
    loss.backward()
    # Now perform our descent method using the gradients
    optimiser.step()

    return optimiser


def calculate_scores(batch_data, standard_data, run_id_str):
    """
    Calculate the scores associated with each run.
    Args:
        batch_data: The large data dictionary containing all solve information
        standard_data: The data related to our instances from solving under normal conditions
        run_id_str: The identifier string of the run in the larger scale of training. Used for our output

    Returns:
        A dictionary for each score metric, showing how for each instance, seed, and cut-sel param sample, the
        relative improvement compared to the baseline solve did
    """

    def get_scores_by_metric(metric, lower_is_better=True, difference=True):
        """
        Function that is used to get scores per individual run over different metrics.
        Args:
            metric: The metric we are interested in: 'eg: gap'. This needs to be a key of the .yml produced by
            Slurm/solve_instance_seed.py which was generated when first calling generate_standard_solve_info.py
            lower_is_better: Whether the metric should be rewarded for being lower or higher. I.e. a smaller gap is good
            difference: Whether or not the result should be compared to standard_solve_info. We set False for cut params
        Returns:
            The percentage improvement of the metric compared to the run under default parameters
        """

        # The main scores dictionary
        scores = {}

        for instance in batch_data:
            scores[instance] = {}
            for rand_seed in batch_data[instance]:
                scores[instance][rand_seed] = []
                for sample_i in batch_data[instance][rand_seed]:
                    # Make a quick short-cut to access the data easily
                    sd = standard_data[instance][rand_seed]
                    bd = batch_data[instance][rand_seed][sample_i]

                    # Add the relative improvements in all measures for a specific instance and seed.
                    # Note: If difference is set to False then we simply return the flat value. This should be debug
                    if not difference:
                        scores[instance][rand_seed].append(bd[metric])
                    elif lower_is_better:
                        scores[instance][rand_seed].append((sd[metric] - bd[metric]) / (abs(sd[metric]) + 1e-8))
                    else:
                        scores[instance][rand_seed].append((bd[metric] - sd[metric]) / (abs(sd[metric]) + 1e-8))
                    # We check if any value has abs > 1. This is not necessarily an error, but helps check our eps works
                    if abs(scores[instance][rand_seed][-1]) > 1 and difference:
                        # Make sure the abs value doesn't go above 1. Too large scores distort values.
                        # This isn't a bug. (0.1 / (0+1e-8)) as an example for a valid gap calculation
                        # Make the output go to -1 or 1 depending on the sign
                        scores[instance][rand_seed][-1] /= abs(scores[instance][rand_seed][-1])
        # For debugging purposes, output the scores over each instance and metric
        # print('{} - {} : {}'.format(run_id_str, metric, scores), flush=True)

        return scores

    # We use the following measures: dual_bound, primal_bound, gap, lp_iterations, num_nodes, num_cuts, solve_time,
    # solution_fractionality, primal_dual_integral

    # Get the dual_bound scores
    dual_bound_scores = get_scores_by_metric('dual_bound', lower_is_better=False)
    # Get the primal bound scores
    primal_bound_scores = get_scores_by_metric('primal_bound')
    # Get the gap scores
    gap_scores = get_scores_by_metric('gap')
    # Get the number of LP iteration scores
    lp_iteration_scores = get_scores_by_metric('num_lp_iterations')
    # Get the number of nodes in the branch and bound tree
    num_node_scores = get_scores_by_metric('num_nodes')
    # Get the number of cuts applied to the problem
    num_cut_scores = get_scores_by_metric('num_cuts', lower_is_better=False)
    # Get the solve_time
    solve_time_scores = get_scores_by_metric('solve_time')
    # Get the solution fractionality
    sol_fractionality_scores = get_scores_by_metric('solution_fractionality')
    # Get the primal dual integral scores
    primal_dual_integral_scores = get_scores_by_metric('primal_dual_integral')

    # Now create debug statements for the cut-selector params. We are not interested in their improvement, just values
    for cut_sel_param in ['dir_cut_off', 'efficacy', 'int_support', 'obj_parallelism']:
        _ = get_scores_by_metric(cut_sel_param, difference=False)

    return (gap_scores, gap_scores, dual_bound_scores, primal_bound_scores, lp_iteration_scores,
            num_node_scores, num_cut_scores, solve_time_scores, sol_fractionality_scores, primal_dual_integral_scores)


def submit_slurm_jobs(data_dir, temp_dir, outfile_dir, num_samples, batch_instances, rand_seeds,
                      time_limit=-1, root=True, exclusive=False):
    """
    Submits slurm jobs of the entire batch. Each job consists of an instance being run with cut-selector
    parameters produced by a perturbed GNN.
    Args:
        data_dir: The directory containing all instances and data files for feature vectors and graph representations
        temp_dir: The directory in which all temporary files per batch will be dumped then deleted (e.g. cut-sel params)
        outfile_dir: The directory in which the slurm out-files for individual jobs will be dumped
        num_samples: The number of sampled cut-sel params files to generate per instance and seed pairing
        batch_instances: The instance names of our .mps instances
        rand_seeds: The random seeds for our SCIP random seed shift
        time_limit: The time limit for each SCIP instance of the slurm jobs. -1 means no time-limit
        root: Whether the SCIP instance for each slurm job should only be solved at the root node
        exclusive: Whether the jobs should take an entire node each. This makes the timing of the job reproducible

    Returns:
        Nothing. All results from the slurm jobs should be output into individual files
    """
    assert type(time_limit) == int

    # Get the list of slurm job IDs that our jobs use. We need these IDs to make our final checker job
    slurm_job_ids = []

    # The main loop that cycles over samples-instances-seeds and produces a job for each
    for sample_i in range(num_samples):
        for instance_i, instance in enumerate(batch_instances):
            time.sleep(0.2)
            for rand_seed in rand_seeds:
                ji = run_python_slurm_job(python_file='Slurm/solve_instance_seed_noise.py',
                                          job_name='{}--{}--{}'.format(instance, rand_seed, sample_i),
                                          outfile=os.path.join(outfile_dir, '%j__{}__{}__{}.out'.format(
                                              instance, rand_seed, sample_i)),
                                          time_limit=120,
                                          arg_list=[temp_dir,
                                                    get_filename(data_dir, instance, rand_seed, trans=True, root=False,
                                                                 sample_i=None, ext='mps'),
                                                    instance, rand_seed, sample_i, time_limit, root, False, False],
                                          exclusive=exclusive)
                slurm_job_ids.append(ji)

    # Now submit the checker job that has dependencies slurm_job_ids
    ji = run_python_slurm_job(python_file='Slurm/safety_check.py',
                              job_name='cleaner',
                              outfile=os.path.join(outfile_dir, 'a--a--a.out'),
                              time_limit=10,
                              arg_list=[os.path.join(temp_dir, 'cleaner.txt')],
                              dependencies=slurm_job_ids)

    return 'cleaner.txt'


def wait_for_slurm_jobs_and_extract_solve_info(temp_dir, batch_instances, rand_seeds, num_samples,
                                               sampled_cut_selector_params, signal_file, root=True, wait_time=3):
    """
    Function that puts the program to sleep until all slurm jobs are complete. Once all jobs are complete
    it extract the information from the individual YAML files produced by each run.
    To see if all jobs are complete, it checks if the signal_file has been created. This is created from a job
    that will only run if all other jobs in the batch have been completed.
    Args:
        temp_dir: The temporary file directory where all files associated with the current batch are stored
        batch_instances: A list containing all instance names of the batch
        rand_seeds: The random seeds which we used to in parallel SCIP solves
        num_samples: The number of samples which we took for each instance-seed pairing
        sampled_cut_selector_params: The dictionary containing all the cut-selector params used in our runs
        signal_file: The file that when created indicates all jobs are complete
        root: Whether the solve info we're waiting on was restricted to the root node. This affects file naming.
        wait_time: The wait_time between asking slurm about the job statuses
    Returns:
        All solve information related to the current batch
    """

    # Put the program to sleep until all of slurm jobs are complete
    time.sleep(wait_time)
    while signal_file not in os.listdir(temp_dir):
        time.sleep(wait_time)

    # Initialise the data directory which we'll load all of our solve information into
    data = {instance: {} for instance in batch_instances}

    # Check if there were problems with any instance solving. All successful instances should have a YAML file
    for instance in batch_instances:
        for rand_seed in rand_seeds:
            # Keep track of all runs that failed, regardless of the reason. We then remove those samples
            invalid_runs = []
            for sample_i in range(num_samples):
                # Get the yml file name that would represent this run
                file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=sample_i,
                                         ext='yml')
                if check_instance_solved(temp_dir, instance, [rand_seed], sample_i, root):
                    # Add the key to the dictionary if it doesn't yet exist
                    if rand_seed not in data[instance]:
                        data[instance][rand_seed] = {}
                    # Now load in the YAML information from the solve
                    with open(file_name, 'r') as s:
                        data[instance][rand_seed][sample_i] = yaml.safe_load(s)
                else:
                    logging.warning('No file {} found from its run'.format(file_name))
                    invalid_runs.append(sample_i)
            # Now remove the invalid run parameters from our sampled parameters
            for invalid_i in range(len(invalid_runs)):
                sampled_cut_selector_params[instance][rand_seed].pop(invalid_runs[invalid_i] - invalid_i)

    return data, sampled_cut_selector_params


def check_instance_solved(temp_dir, instance, rand_seeds, sample_i, root):
    """
    Checks if instances for a specific sample-instance-[list of seed] pairing solved successfully
    Args:
        temp_dir: The directory containing all batch related solve info (temporary files)
        instance: The instance name
        rand_seeds: A list of seeds which we use to solve our instances with
        sample_i: The index of the sample we are interested in
        root: Whether the solve was restricted to the root-node. This affects file naming

    Returns:
        True if run was successful else False
    """

    # Initialise the variable saying whether our runs were successful
    instance_successful = True

    # Cycle over the rand_seeds and check if the solve_info exists
    for rand_seed in rand_seeds:
        file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=sample_i, ext='yml')
        # If the yml doesn't exist then log that their was an error with the run
        if not os.path.isfile(file_name):
            instance_successful = False
            logging.warning('Failed to find yml file for instance {} seed {} sample {} in dir {}'.format(
                instance, rand_seed, sample_i, temp_dir))

    return instance_successful


def generate_batches(instances, rel_batch_size, random_state):
    """
    Creates batches of the instances so that each batch is similar to rel_batch_size.
    The result is a list of lists, with the instance names divided into batches
    Args:
        instances: A list containing all names of the instances
        rel_batch_size: The relative batch size with respect to the total number of instances
        random_state: Random state for ensuring that the shuffle mechanism is duplicable.

    Returns:
        A list of batches for the instances
    """

    assert len(instances) > 0
    assert 0 < rel_batch_size <= 1

    # Get the number of batches from the relative batch size
    num_batches = round(1 / rel_batch_size)
    if len(instances) < num_batches:
        logging.error('rel_batch_size {} results in {} many batches when there are only {} many instances.'
                      'Changed to {} many batches'.format(rel_batch_size, num_batches, len(instances), len(instances)))
        num_batches = len(instances)

    # Shuffle the instances in the same way as our RandomState. This way we do not always get the same batches
    if random_state is not None:
        shuffle_order = random_state.permutation(len(instances))
    else:
        shuffle_order = np.random.permutation(len(instances))
    instances = [instance for _, instance in sorted(zip(shuffle_order, instances))]

    # Create the batches, ensuring that all batches have similar size.
    batch_instances = [[] for _ in range(num_batches)]
    for i, instance in enumerate(instances):
        batch_instances[i % num_batches].append(instance)

    return batch_instances, random_state


def get_instances(data_dir):
    """
    Retrieves a list of the names for the instances
    Args:
        data_dir: The directory containing our instance data

    Returns:
        A list of instance base_names (i.e toll-like), explicitly not (toll-like__seed_7.mps)
    """
    assert os.path.isdir(data_dir)
    # Initialise the list of instances a set. We use a set as our file naming system has multiple files per instance
    instances = set()
    for file in os.listdir(data_dir):
        if file.endswith('.yml'):
            # Extract the base-name of the instances from the file
            instances.add(os.path.splitext(os.path.splitext(file)[0])[0].split('__')[0])
    return sorted(list(instances))


def get_standard_solve_data(data_dir, root=True):
    """
    Function for getting all standard solve data
    Args:
        data_dir: The data directory containing all standard solve info and feature vector information
        root: A kwarg that indicates whether root solve or non node limit restricted solve should be retrieved

    Returns:
        A dictionary containing all data related to solving instances under standard conditions.
    """

    # First get all files in our data_directory
    files = os.listdir(data_dir)

    # We're interested in the .yml files produced by generate_standard_solve_info.py
    solve_info_files = [file for file in files if file.endswith('.yml')]
    # Filter out the yml files that are either restricted to the root node or not
    solve_info_files = [file for file in solve_info_files if ('__root__' in file and root) or
                        ('__root__' not in file and not root)]

    # Initialise the standard solve info dictionary
    standard_solve_data = {}

    # Now cycle through all YAML files. They should have the form {instance}__seed__{rand_seed}.yml
    for file in solve_info_files:

        file_base_name = os.path.splitext(file)[0]

        # Using the file naming pattern extract the instance and rand_seed
        instance, rand_seed = file_base_name.split('__')[0], file_base_name.split('__')[-1]
        assert rand_seed.isdigit(), 'Random seed {} used to generate file {} is not an int'.format(rand_seed, file)
        rand_seed = int(rand_seed)

        # If this is the first time seeing the instance make sure it has a sub-dictionary
        if instance not in standard_solve_data:
            standard_solve_data[instance] = {}

        # Load the YAML file as a dictionary into our data structure
        with open(os.path.join(data_dir, file), 'r') as s:
            standard_solve_data[instance][rand_seed] = yaml.safe_load(s)

    return standard_solve_data


def get_rand_seeds_from_feature_generators(standard_solve_data):
    """
    Function for getting the random seeds we used to generate our features and standard solves with.
    Args:
        standard_solve_data: The dictionary containing all information generated by generate_standard_solve_info.py

    Returns:
        The random seeds used in previous data creation
    """

    # Our dictionary structure is standard_solve_info[instance][rand_seed] = {solve_info}. Extract a single instance
    random_instance = list(standard_solve_data.keys())[0]

    # Now check that each instance has exactly the same random_seeds
    for rand_seed in standard_solve_data[random_instance].keys():
        for instance in standard_solve_data.keys():
            assert rand_seed in standard_solve_data[instance], 'Instance {} was not found with rand_seed {}'.format(
                instance, rand_seed)

    # Now check that the random seeds represent a standard range() in python
    rand_seeds = sorted(list(standard_solve_data[random_instance].keys()))
    for rand_seed_i, rand_seed in enumerate(rand_seeds):
        if rand_seed_i < len(rand_seeds) - 1:
            assert rand_seed == rand_seeds[
                rand_seed_i + 1] - 1, 'Random seeds {} do not represent a python range'.format(rand_seeds)

    return rand_seeds


def create_tensorboard_writer(seed_init, run_dir, instance=None):
    """
    Function to create a tensorboard SummaryWriter object. This object will store all run data, and be later used
    to visualise results of runs.
    Args:
        seed_init: The seed_init used in the run. This is for reproducibility
        run_dir: The directory in which all tensorboard run information will be stored
        instance: The instance this run pertains to. It affects the naming convention
    Returns:
        The tensorboard SummaryWriter object
    """

    instance_prefix = (instance + '_') if instance is not None else ''
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if seed_init >= 0:
        log_dir = os.path.join(run_dir, instance_prefix + current_time + '_' + '{}'.format(seed_init))
    else:
        log_dir = os.path.join(run_dir, instance_prefix + current_time)

    return SummaryWriter(log_dir=log_dir)


def add_data_to_tensorboard_writer(tensorboard_writer, batch_data, scores, gaps, dual_bounds, primal_bounds,
                                   lp_iterations, num_nodes, num_cuts, sol_times, sol_fractions,
                                   primal_dual_integrals, run_i, neural_network, test=False):
    """
    Function that adds all relevant data to the tensorboard summary writer.
    The tensorboard summary writer data should be found in run_dir
    Args:
        tensorboard_writer: The SummaryWriter object
        batch_data: The complete data associated with the runs from the batch
        scores: The scores that will be ranked and used to update the neural_network
        gaps: The scores if gap was the measurement we ranked by
        dual_bounds: The scores if dual_bound improvement was the measurement we ranked by
        primal_bounds: The scores if primal_bound improvement was the measurement we ranked by
        lp_iterations: The scores if num_lp_iterations was the measurement we ranked by
        num_nodes: The scores if num_nodes was the measurement we ranked by
        num_cuts: The scores if num_cuts was the measurement we ranked by
        sol_times: The scores if solve_times was the measurement we ranked by
        sol_fractions: The scores if sol_fractionality was the measurement we ranked by
        primal_dual_integrals: The scores if the primal-dual-integral was the measurement we ranked by
        run_i: The index of this batch in the larger scale of the run
        neural_network: The neural_network object
        test: A boolean on whether the data added to tensorboard should be preceded by test

    Returns:
        The tensorboard SummaryWriter object
    """

    def get_mean_median_and_std_from_dictionary(data, metric=None):
        values = []
        for instance in data:
            for rand_seed in data[instance]:
                if type(data[instance][rand_seed]) == dict:
                    samples = list(data[instance][rand_seed].keys())
                else:
                    samples = range(len(data[instance][rand_seed]))
                for sample_i in samples:
                    if metric is not None:
                        values.append(data[instance][rand_seed][sample_i][metric])
                    else:
                        values.append(data[instance][rand_seed][sample_i])
        if len(values) <= 0:
            return 0, 0, 0
        return np.mean(values), np.median(values), np.std(values)

    # Add the cut-selector params. We get this information from the batch_data
    for cut_sel_param in ['dir_cut_off', 'efficacy', 'int_support', 'obj_parallelism']:
        # Add the average value and standard deviation to the SummaryWriter
        mean, median, std = get_mean_median_and_std_from_dictionary(batch_data, metric=cut_sel_param)
        if test:
            tensorboard_writer.add_scalar('test_' + cut_sel_param, mean, run_i)
            tensorboard_writer.add_scalar('test_' + cut_sel_param + '_median', median, run_i)
            tensorboard_writer.add_scalar('test_' + cut_sel_param + '_std', std, run_i)
        else:
            tensorboard_writer.add_scalar(cut_sel_param, mean, run_i)
            tensorboard_writer.add_scalar(cut_sel_param + '_median', median, run_i)
            tensorboard_writer.add_scalar(cut_sel_param + '_std', std, run_i)

    # Add the alternative scoring methods as well. These can be used to spot patterns
    alternate_scores = [('scores', scores), ('gap', gaps), ('db', dual_bounds), ('pb', primal_bounds),
                        ('lp_iter', lp_iterations), ('num_nodes', num_nodes), ('num_cuts', num_cuts),
                        ('solve_time', sol_times), ('sol_frac', sol_fractions), ('pd_integral', primal_dual_integrals)]
    for alternate_score in alternate_scores:
        mean, median, std = get_mean_median_and_std_from_dictionary(alternate_score[1])
        if test:
            tensorboard_writer.add_scalar('test_' + alternate_score[0], mean, run_i)
            tensorboard_writer.add_scalar('test_' + alternate_score[0] + '_median', median, run_i)
            tensorboard_writer.add_scalar('test_' + alternate_score[0] + '_std', std, run_i)
        else:
            tensorboard_writer.add_scalar(alternate_score[0], mean, run_i)
            tensorboard_writer.add_scalar(alternate_score[0] + '_median', median, run_i)
            tensorboard_writer.add_scalar(alternate_score[0] + '_std', std, run_i)

    return tensorboard_writer


def run_test_set(data_dir, temp_dir, outfile_dir, neural_network, batch_instances, rand_seeds, standard_solve_data,
                 test_i, run_i, batches_per_test, tensorboard_writer, time_limit=-1, root=True, rm_temp_files=True,
                 exclusive=False, create_yaml=False):
    """
    Function for running the test set. It uses existing functions that we would normally train with, but simply
    generates a single dummy sample file containing the exact output of the GNN, and does not call the function to
    update neural_network.
    Args:
        data_dir: The directory containing our data
        temp_dir: The directory containing all temporary files. We will output into here
        outfile_dir: The directory in which we will dump our slurm .out files
        neural_network: The torch neural_network object
        batch_instances: The instances that we will be using in he batch
        rand_seeds: The random seeds that we use over our entire run
        standard_solve_data: The dictionary containing our standard solve data under normal conditions
        test_i: The index of our test relative to other tests
        run_i: The index of the current run relative to the other runs
        batches_per_test: How many batches should be run before a test is run
        tensorboard_writer: The tensorboard SummaryWriter object
        time_limit: How long the SCIP instance should run for. -1 means no limit will be applied
        root: A boolean that says whether node_limit should be 1 or -1
        rm_temp_files: Should the temporary files produced by each run be deleted
        exclusive: Whether the runs for this test set should take a complete node each. This makes the time reproducible
        create_yaml: If a YAML file containing average scores and cut-sel parameters should be created

    Returns:
        The new index of the next test provided that a test was run and the tensorboard object
    """

    if run_i % batches_per_test != batches_per_test - 1:
        return test_i, tensorboard_writer

    # Create a slurm outfile directory for the test run
    test_outfile_dir = os.path.join(outfile_dir, 'test_' + str(test_i))
    os.mkdir(test_outfile_dir)

    # Set the neural network into eval mode.
    neural_network.eval()

    # Create our cut-selector parameter sample values / files as well as the distributions they come from
    neural_network, sampled_cut_params, multivariate_normals = create_cut_selector_params(data_dir, temp_dir,
                                                                                          neural_network,
                                                                                          batch_instances, rand_seeds,
                                                                                          0, 1, 1)

    # Set the neural network back into training mode
    neural_network.train()

    # Create slurm jobs. Each job consists of one instance-seed-sample pairing
    signal_file = submit_slurm_jobs(data_dir, temp_dir, test_outfile_dir, 1, batch_instances, rand_seeds, time_limit,
                                    root, exclusive=exclusive)

    # Wait for slurm jobs to finish and read in all solve information from the jobs
    batch_data, sampled_cut_params = wait_for_slurm_jobs_and_extract_solve_info(temp_dir, batch_instances,
                                                                                rand_seeds, 1, sampled_cut_params,
                                                                                signal_file, root=root)

    # Generate scores based on solve_information
    (scores, gaps, dual_bounds, primal_bounds, lp_iters, num_nodes, num_cuts, sol_times, sol_fracs,
     primal_dual_ints) = calculate_scores(batch_data, standard_solve_data, 'test_' + str(test_i))

    # Add all data related to the batch to the summary writer
    tensorboard_writer = add_data_to_tensorboard_writer(tensorboard_writer, batch_data, scores, gaps,
                                                        dual_bounds, primal_bounds,
                                                        lp_iters, num_nodes, num_cuts, sol_times,
                                                        sol_fracs, primal_dual_ints, test_i, neural_network, test=True)

    # Create a YAML file with average scores and cut-sel params per instance if we have this flag
    if create_yaml:
        bd = batch_data
        for instance in batch_data:
            dir_cut_off = float(np.mean([bd[instance][rand_seed][0]['dir_cut_off'] for rand_seed in rand_seeds]))
            efficacy = float(np.mean([bd[instance][rand_seed][0]['efficacy'] for rand_seed in rand_seeds]))
            int_support = float(np.mean([bd[instance][rand_seed][0]['int_support'] for rand_seed in rand_seeds]))
            obj_parallel = float(np.mean([bd[instance][rand_seed][0]['obj_parallelism'] for rand_seed in rand_seeds]))
            score = float(np.mean([scores[instance][rand_seed][0] for rand_seed in rand_seeds]))
            dual_bound = float(np.mean([dual_bounds[instance][rand_seed][0] for rand_seed in rand_seeds]))
            lp_iter = float(np.mean([lp_iters[instance][rand_seed][0] for rand_seed in rand_seeds]))
            num_cut = float(np.mean([num_cuts[instance][rand_seed][0] for rand_seed in rand_seeds]))
            sol_frac = float(np.mean([sol_fracs[instance][rand_seed][0] for rand_seed in rand_seeds]))
            gap = float(np.mean([gaps[instance][rand_seed][0] for rand_seed in rand_seeds]))
            parameters = []
            for rand_seed in rand_seeds:
                parameters.append([bd[instance][rand_seed][0]['dir_cut_off'],
                                   bd[instance][rand_seed][0]['efficacy'],
                                   bd[instance][rand_seed][0]['int_support'],
                                   bd[instance][rand_seed][0]['obj_parallelism']])
            yaml_data = {instance: {'dir_cut_off': dir_cut_off, 'efficacy': efficacy, 'int_support': int_support,
                                    'obj_parallelism': obj_parallel, 'score': score, 'dual_bound': dual_bound,
                                    'gap': gap, 'num_lp_iterations': lp_iter, 'num_cuts': num_cut,
                                    'solution_fractionality': sol_frac, 'parameters': parameters}}
            yaml_file = get_filename(data_dir, instance, None, True, False, None, 'yaml')
            with open(yaml_file, 'w') as s:
                yaml.dump(yaml_data, s)

    if rm_temp_files:
        remove_temp_files(temp_dir)

    return test_i + 1, tensorboard_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=is_dir)
    parser.add_argument('run_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('prev_network', type=str)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('num_epochs', type=int)
    parser.add_argument('rel_batch_size', type=float)
    parser.add_argument('num_samples', type=int)
    parser.add_argument('seed_init', type=int)
    parser.add_argument('one_at_a_time', type=str_to_bool)
    args = parser.parse_args()

    # Remove all slurm .out files produced by previous runs
    args.outfile_dir = os.path.join(args.outfile_dir, 'train_network')
    if not os.path.isdir(args.outfile_dir):
        os.mkdir(args.outfile_dir)
    if not args.one_at_a_time:
        args.outfile_dir = os.path.join(args.outfile_dir, 'full')
        if not os.path.isdir(args.outfile_dir):
            os.mkdir(args.outfile_dir)
        else:
            remove_slurm_files(args.outfile_dir)

    if args.prev_network == 'None':
        args.prev_network = None
    assert args.prev_network is None or os.path.isfile(args.prev_network), args.prev_network

    # The one_at_a_time arg tells us that we want to train a single instance at a time.
    if args.one_at_a_time:
        instance_names = get_instances(args.data_dir)
        for instance_name in instance_names:
            instance_outfile_dir = os.path.join(args.outfile_dir, instance_name)
            if not os.path.isdir(instance_outfile_dir):
                os.mkdir(os.path.join(args.outfile_dir, instance_name))
            else:
                remove_slurm_files(instance_outfile_dir)
            print('Training Instance {}'.format(instance_name), flush=True)
            train_network(args.data_dir, args.run_dir, args.temp_dir, args.prev_network,
                          instance_outfile_dir, args.num_epochs, args.rel_batch_size,
                          args.num_samples, args.seed_init, single_instance=instance_name)
    else:
        # The main function call to train a network from scratch
        train_network(args.data_dir, args.run_dir, args.temp_dir, args.prev_network, args.outfile_dir, args.num_epochs,
                      args.rel_batch_size, args.num_samples, args.seed_init)
