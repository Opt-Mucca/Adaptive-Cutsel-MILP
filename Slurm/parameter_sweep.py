#! /usr/bin/env python
import os
import argparse
import yaml
import numpy as np
from utilities import remove_instance_solve_data, is_dir
from utilities import str_to_bool, remove_slurm_files, get_filename, remove_temp_files, run_python_slurm_job
from Slurm.train_neural_network import wait_for_slurm_jobs_and_extract_solve_info


def parameter_sweep(data_dir, temp_dir, outfile_dir, instance, rand_seeds, root):
    """
    Main function for doing a parameter sweep and finding the potential improvement through adaptive cut-selection.
    This parameter sweep goes through all convex combinations of dir_cut_off, efficacy, int_support, and
    obj_parallelism. All values are a multiple of 0.1, and they must sum to 1.
    Args:
        data_dir: The directory containing all of our standard generated data
        temp_dir: The directory where we will throw all of our temporary files only related to this run
        outfile_dir: The directory where we store all .log files for this run and runs called from it
        instance: The instance that we perform our parameter sweep on
        rand_seeds: The list of random seeds that we'll be using
        root: Whether our solve is restricted to the root node or not. (Warning: Takes a long time if not)

    Returns:
        The average improvement, the convex combinations that gave this improvement, and amount of combinations tried
    """

    # Create a new outfile dir
    assert not os.path.isdir(os.path.join(outfile_dir, instance))
    outfile_dir = os.path.join(outfile_dir, instance)
    os.mkdir(outfile_dir)

    # Make all the convex combinations for the cut-selector parameters
    convex_combinations = []
    # Our convex combinations are sum_i(lambda_i) == 1, where lambda_i = a * 0.1
    for dir_cut_off in range(0, 11):
        for efficacy in range(0, 11 - dir_cut_off):
            for int_support in range(0, 11 - dir_cut_off - efficacy):
                obj_parallelism = 10 - dir_cut_off - efficacy - int_support
                convex_combinations.append([dir_cut_off / 10, efficacy / 10, int_support / 10, obj_parallelism / 10])

    # Create the cut-selector .npy files with sample_i they're index
    for rand_seed in rand_seeds:
        for sample_i, sample in enumerate(convex_combinations):
            file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=sample_i,
                                     ext='npy')
            np.save(file_name, np.array(convex_combinations[sample_i]))

    # Start the jobs
    slurm_job_ids = []
    for rand_seed in rand_seeds:
        for sample_i in range(len(convex_combinations)):
            instance_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None,
                                         ext='mps')
            ji = run_python_slurm_job(python_file='Slurm/solve_instance_seed_noise.py',
                                      job_name='{}--{}--{}'.format(instance, rand_seed, sample_i),
                                      outfile=os.path.join(outfile_dir, '%j__{}__seed__{}__sample__{}.out'.format(
                                       instance, rand_seed, sample_i)),
                                      time_limit=120,
                                      arg_list=[temp_dir, instance_file,
                                                instance, rand_seed, sample_i, -1, root, False, False])
            slurm_job_ids.append(ji)

    # Now submit the checker job that has dependencies slurm_job_ids
    signal_file = 'cleaner.txt'
    _ = run_python_slurm_job(python_file='Slurm/safety_check.py',
                             job_name='cleaner',
                             outfile=os.path.join(outfile_dir, 'a--a--a.out'),
                             time_limit=10,
                             arg_list=[os.path.join(temp_dir, signal_file)],
                             dependencies=slurm_job_ids)

    # Wait on jobs to finish and extract the solve information
    sampled_cut_selector_params = {instance: {rand_seed: convex_combinations for rand_seed in rand_seeds}}
    data, _ = wait_for_slurm_jobs_and_extract_solve_info(temp_dir, [instance], rand_seeds, len(convex_combinations),
                                                         sampled_cut_selector_params, signal_file, root=True,
                                                         wait_time=10)

    # Now average our results over each random seed
    mean_scores = {}
    for sample_i in range(len(convex_combinations)):
        scores = [data[instance][rand_seed][sample_i]['gap'] for rand_seed in rand_seeds]
        if len(scores) > 0:
            mean_scores[sample_i] = np.mean(scores)

    # We want to then take the best score over the samples
    best_combination_indices = [None]
    for sample_i in range(len(convex_combinations)):
        if best_combination_indices[0] is None or mean_scores[sample_i] < mean_scores[best_combination_indices[0]]:
            best_combination_indices = [sample_i]
        elif mean_scores[sample_i] == mean_scores[best_combination_indices[0]]:
            best_combination_indices.append(sample_i)

    assert best_combination_indices[0] is not None, 'Instance {} has no best performing score with dict ' \
                                                    '{}'.format(instance, mean_scores)

    # We also want to take the worst scores over the samples. This is so we can filter instances.
    worst_combination_indices = [None]
    for sample_i in range(len(convex_combinations)):
        if worst_combination_indices[0] is None or mean_scores[sample_i] > mean_scores[worst_combination_indices[0]]:
            worst_combination_indices = [sample_i]
        elif mean_scores[sample_i] == mean_scores[worst_combination_indices[0]]:
            worst_combination_indices.append(sample_i)

    # Now average the results for the standard solve over the random seeds
    standard_scores = []
    for rand_seed in rand_seeds:
        yml_file = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='yml')
        with open(yml_file, 'r') as s:
            info = yaml.safe_load(s)
        score = info['gap']
        standard_scores.append(score)
    standard_score = np.mean(standard_scores)

    # Now compare against default SCIP values
    best_score = mean_scores[best_combination_indices[0]]
    worst_score = mean_scores[worst_combination_indices[0]]
    # TODO: Depending on if we use GAP or DB the direction of improvement changes.
    best_improvement = float((standard_score - best_score) / (np.abs(standard_score) + 1e-8))
    worst_improvement = float((standard_score - worst_score) / (np.abs(standard_score) + 1e-8))

    # Get all convex combinations from their indices
    best_combinations = [convex_combinations[best_i] for best_i in best_combination_indices]
    print('Instance {} improved by {} using {} parameters'.format(instance, best_improvement, len(best_combinations)),
          flush=True)

    is_potential_improvement = True
    unique_optimal_choices = True
    if best_improvement - worst_improvement < 0.001:
        print('{} has best improvement {} vs worst improvement {}. Filtering instance.'.format(instance,
                                                                                               best_improvement,
                                                                                               worst_improvement),
              flush=True)
        is_potential_improvement = False
    if len(best_combinations) >= (1/4) * len(convex_combinations):
        print('{} has >=1/4 of cut-sel param combinations as optimal choice. Filtering instance'.format(instance),
              flush=True)
        unique_optimal_choices = False

    if not is_potential_improvement or not unique_optimal_choices:
        return None, None, is_potential_improvement, unique_optimal_choices
    else:
        return best_improvement, best_combinations, is_potential_improvement, unique_optimal_choices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('root', type=str_to_bool)
    args = parser.parse_args()

    # Remove all solve information from previous runs
    remove_temp_files(args.temp_dir)
    args.outfile_dir = os.path.join(args.outfile_dir, 'grid_search')
    if not os.path.isdir(args.outfile_dir):
        os.mkdir(args.outfile_dir)
    else:
        remove_slurm_files(args.outfile_dir)

    # Initialise a list of instances
    instance_names = set()
    random_seeds = set()

    for file in os.listdir(args.data_dir):
        # Extract the instance and seed from the instance:path
        if file.endswith('.yml'):
            instance_name = os.path.splitext(file)[0].split('__')[0]
            random_seed = int(os.path.splitext(file)[0].split('__')[-1])
            instance_names.add(instance_name)
            random_seeds.add(random_seed)

    instance_names = list(instance_names)
    random_seeds = list(random_seeds)

    valid_instances = []
    no_improvement_instances = set()
    always_improvement_instances = set()

    # Initialise the dictionary where we store the potential improvement for the instance
    improvements = {instance_name: {} for instance_name in instance_names}

    for instance_name in instance_names:
        # The main function call to begin the parameter sweep
        gain, parameters, is_improvement, is_unique = parameter_sweep(args.data_dir, args.temp_dir, args.outfile_dir,
                                                                      instance_name, random_seeds, args.root)
        # We don't want any instances in our training set where all cut-sel param choices result in same solve process
        if not is_improvement or not is_unique:
            remove_instance_solve_data(args.data_dir, instance_name, suppress_warnings=True)
            del improvements[instance_name]
            if not is_improvement:
                no_improvement_instances.add(instance_name)
            if not is_unique:
                always_improvement_instances.add(instance_name)
        else:
            improvements[instance_name]['improvement'] = gain
            improvements[instance_name]['parameters'] = parameters
            valid_instances.append(instance_name)
        # Remove the temp files produced by the previous run
        remove_temp_files(args.temp_dir)

    print('{} instances remain from {}'.format(len(valid_instances), len(instance_names)), flush=True)
    print('{} instances filtered as no improvements possible. Instances {}'.format(
        len(no_improvement_instances), no_improvement_instances), flush=True)
    print('{} instances filtered as too many optimal parameter choices. Instances {}'.format(
        len(always_improvement_instances), always_improvement_instances), flush=True)
    filtered_intersection = always_improvement_instances.intersection(no_improvement_instances)
    print('{} many instances in overlap of filtering. Instances are {}'.format(
        len(filtered_intersection), filtered_intersection), flush=True)

    # Dump the yml file containing all of our solve info into the right place. Use .YAML instead
    yaml_file = os.path.join(args.data_dir, 'potential_improvements.yaml')
    with open(yaml_file, 'w') as ss:
        yaml.dump(improvements, ss)
    print('Average Improvement: {}'.format(np.mean([improvements[i]['improvement'] for i in valid_instances])))
