#! /usr/bin/env python
import os
import argparse
import yaml
import numpy as np
import logging
import time
from utilities import remove_instance_solve_data, is_dir, get_instances, get_random_seeds, get_slurm_output_file, \
    str_to_bool, remove_slurm_files, get_filename, remove_temp_files, run_python_slurm_job
import parameters


def parameter_sweep(instance_dir, solution_dir, default_results_dir, temp_dir, outfile_dir, instance, rand_seeds, root):
    """
    Main function for doing a parameter sweep and finding the potential improvement through adaptive cut-selection.
    This parameter sweep goes through all convex combinations of dir_cut_off, efficacy, int_support, and
    obj_parallelism. All values are a multiple of 0.1, and they must sum to 1.
    Args:
        instance_dir: The directory containing all of our presolved MILP instances
        solution_dir: The directory containing all our primal solutions for the presolved MILP instances
        default_results_dir: The directory containing all default SCIP run statistics for each instance
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
            instance_file = get_filename(instance_dir, instance, rand_seed, trans=True, root=False, sample_i=None,
                                         ext='mps')
            solution_file = get_filename(solution_dir, instance, rand_seed, trans=True, root=False, sample_i=None,
                                         ext='sol')
            ji = run_python_slurm_job(python_file='Slurm/solve_instance_seed_noise.py',
                                      job_name='{}--{}--{}'.format(instance, rand_seed, sample_i),
                                      outfile=os.path.join(outfile_dir, '%j__{}__seed__{}__sample__{}.out'.format(
                                       instance, rand_seed, sample_i)),
                                      time_limit=120,
                                      arg_list=[temp_dir, instance_file, solution_file,
                                                instance, rand_seed, sample_i, -1, root, True, False, False])
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
    data, numeric_issue = wait_for_slurm_jobs_and_extract_solve_info(temp_dir, outfile_dir, instance, rand_seeds,
                                                                     len(convex_combinations), signal_file, root=True,
                                                                     wait_time=10)

    if numeric_issue:
        return None, None, True, True, False, False, True, False, None

    # Filter out instances with the same reasons as the standard solve!
    incorrect_cut_amounts = False
    instance_is_optimal = False
    low_primal_dual_difference = False
    if root:
        for rand_seed in rand_seeds:
            for sample_i in range(len(convex_combinations)):
                if data[rand_seed][sample_i]['status'] == 'optimal':
                    instance_is_optimal = True
                if data[rand_seed][sample_i]['primal_dual_difference'] < parameters.MIN_PRIMAL_DUAL_DIFFERENCE:
                    low_primal_dual_difference = True
                min_cuts = parameters.MIN_NUM_CUT_RATIO * parameters.NUM_CUT_ROUNDS * parameters.NUM_CUTS_PER_ROUND
                max_cuts = parameters.MAX_NUM_CUT_RATIO * parameters.NUM_CUT_ROUNDS * parameters.NUM_CUTS_PER_ROUND
                if data[rand_seed][sample_i]['num_cuts'] < min_cuts:
                    incorrect_cut_amounts = True
                elif data[rand_seed][sample_i]['num_cuts'] > max_cuts:
                    incorrect_cut_amounts = True

    # Now average our results over each random seed
    mean_scores = {}
    for sample_i in range(len(convex_combinations)):
        scores = [data[rand_seed][sample_i]['primal_dual_difference'] for rand_seed in rand_seeds]
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
        yml_file = get_filename(default_results_dir, instance, rand_seed, trans=True, root=root, sample_i=None,
                                ext='yml')
        with open(yml_file, 'r') as s:
            info = yaml.safe_load(s)
        score = info['primal_dual_difference']
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

    results_dict = {sample_i: {'dir_cut_off': convex_combinations[sample_i][0],
                               'efficacy': convex_combinations[sample_i][1],
                               'int_support': convex_combinations[sample_i][2],
                               'obj_parallelism': convex_combinations[sample_i][3],
                               'improvement': float((standard_score - mean_scores[sample_i]) /
                                                    (np.abs(standard_score) + 1e-8))}
                    for sample_i in range(len(convex_combinations)) if sample_i in mean_scores}

    if not is_potential_improvement or not unique_optimal_choices:
        return None, None, is_potential_improvement, unique_optimal_choices, incorrect_cut_amounts, \
               instance_is_optimal, numeric_issue, low_primal_dual_difference, results_dict
    else:
        return best_improvement, best_combinations, is_potential_improvement, unique_optimal_choices, \
               incorrect_cut_amounts, instance_is_optimal, numeric_issue, low_primal_dual_difference, results_dict


def wait_for_slurm_jobs_and_extract_solve_info(temp_dir, outfile_dir, instance, rand_seeds, num_samples, signal_file,
                                               root=True, wait_time=3):
    """
    Function that puts the program to sleep until all slurm jobs are complete. Once all jobs are complete
    it extract the information from the individual YAML files produced by each run.
    To see if all jobs are complete, it checks if the signal_file has been created. This is created from a job
    that will only run if all other jobs in the batch have been completed.
    Args:
        temp_dir: The temporary file directory where all files associated with the current batch are stored
        outfile_dir: The directory where all out files for the runs are stored
        instance: The instance we are concerned with
        rand_seeds: The random seeds which we used to in parallel SCIP solves
        num_samples: The number of samples which we took for each instance-seed pairing
        signal_file: The file that when created indicates all jobs are complete
        root: Whether the solve info we're waiting on was restricted to the root node. This affects file naming.
        wait_time: The wait_time between asking slurm about the job statuses
    Returns:
        All solve information related to the current instance. Also tells us if there was an error in any runs
    """

    # Put the program to sleep until all of slurm jobs are complete
    time.sleep(wait_time)
    while signal_file not in os.listdir(temp_dir):
        time.sleep(wait_time)

    # Initialise the data directory which we'll load all of our solve information into
    data = {rand_seed: {} for rand_seed in rand_seeds}
    numeric_issues = False

    # Check if there were problems with the instance during solving. All successful solves should have a YAML file
    for rand_seed in rand_seeds:
        # Keep track of all runs that failed, regardless of the reason. We then remove those samples
        invalid_runs = []
        for sample_i in range(num_samples):
            # Get the yml file name that would represent this run
            file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=sample_i,
                                     ext='yml')
            if not os.path.isfile(file_name):
                logging.warning('Instance {} with seed {} and sample_i {} failed'.format(instance, rand_seed, sample_i))
                # Check if the instance failed for numeric reasons
                out_file = get_slurm_output_file(outfile_dir, instance, rand_seed, sample_i=sample_i)
                with open(out_file, 'r') as s:
                    out_file_contents = s.readlines()
                numeric_issues = False
                for line in out_file_contents:
                    if 'unresolved numerical troubles in LP' in line and '-- aborting' in line:
                        numeric_issues = True
                        break
                if not numeric_issues:
                    print('Instance {} with seed {} and sample {} failed for unknown reasons. Stopping run'.format(
                        instance, rand_seed, sample_i), flush=True)
                else:
                    return None, True
            with open(file_name, 'r') as s:
                data[rand_seed][sample_i] = yaml.safe_load(s)

    return data, numeric_issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=is_dir)
    parser.add_argument('feature_dir', type=is_dir)
    parser.add_argument('default_root_results_dir', type=is_dir)
    parser.add_argument('default_full_results_dir', type=is_dir)
    parser.add_argument('results_dir', type=is_dir)
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
    instance_names = get_instances(args.instance_dir)
    random_seeds = get_random_seeds(args.instance_dir)

    valid_instances = []
    no_improvement_instances = set()
    always_improvement_instances = set()
    incorrect_cut_amount_instances = set()
    optimal_instances = set()
    numerical_issues_instances = set()
    primal_dual_instances = set()

    # Initialise the dictionary where we store the potential improvement for the instance
    improvements = {instance_name: {} for instance_name in instance_names}
    full_results_dict = {instance_name: {} for instance_name in instance_names}

    for instance_name in instance_names:
        # The main function call to begin the parameter sweep
        gain, params, is_improvement, is_unique, no_cuts, optimal, numerics, low_pd_diff, results = parameter_sweep(
            args.instance_dir, args.solution_dir, args.default_root_results_dir, args.temp_dir, args.outfile_dir,
            instance_name, random_seeds, args.root)
        # We don't want any instances in our training set where all cut-sel param choices result in same solve process
        if not is_improvement or not is_unique or no_cuts or optimal or numerics or low_pd_diff:
            remove_instance_solve_data(args.instance_dir, instance_name, suppress_warnings=True)
            remove_instance_solve_data(args.solution_dir, instance_name, suppress_warnings=True)
            remove_instance_solve_data(args.feature_dir, instance_name, suppress_warnings=True)
            remove_instance_solve_data(args.default_root_results_dir, instance_name, suppress_warnings=True)
            remove_instance_solve_data(args.default_full_results_dir, instance_name, suppress_warnings=True)
            del improvements[instance_name]
            del full_results_dict[instance_name]
            if not is_improvement:
                no_improvement_instances.add(instance_name)
            if not is_unique:
                always_improvement_instances.add(instance_name)
            if no_cuts:
                incorrect_cut_amount_instances.add(instance_name)
            if optimal:
                optimal_instances.add(instance_name)
            if numerics:
                numerical_issues_instances.add(instance_name)
            if low_pd_diff:
                primal_dual_instances.add(instance_name)
        else:
            improvements[instance_name]['improvement'] = gain
            improvements[instance_name]['parameters'] = params
            valid_instances.append(instance_name)
            full_results_dict[instance_name] = results

        # Remove the temp files produced by the previous run
        remove_temp_files(args.temp_dir)

    print('{} instances remain from {}'.format(len(valid_instances), len(instance_names)), flush=True)

    # Print out the reasons behind instances being filtered
    print('{} instances filtered as no improvements possible. Instances {}'.format(
        len(no_improvement_instances), no_improvement_instances), flush=True)
    print('{} instances filtered as too many optimal parameter choices. Instances {}'.format(
        len(always_improvement_instances), always_improvement_instances), flush=True)
    print('{} instances filtered for inconsistent amount of cuts. Instances {}'.format(
        len(incorrect_cut_amount_instances), incorrect_cut_amount_instances), flush=True)
    print('{} instances filtered for being root optimal. Instances {}'.format(
        len(optimal_instances), optimal_instances), flush=True)
    print('{} instances filtered for numeric issues. Instances {}'.format(
        len(numerical_issues_instances), numerical_issues_instances), flush=True)
    print('{} instances filtered for low primal dual difference. Instances {}'.format(
        len(primal_dual_instances), primal_dual_instances), flush=True)

    def instance_set_name(idx):
        if idx == 0:
            return 'no improvement possible'
        elif idx == 1:
            return 'too many optimal parameters'
        elif idx == 2:
            return 'root optimal'
        elif idx == 3:
            return 'inconsistent amount of cuts'
        elif idx == 4:
            return 'numerical issues'
        elif idx == 5:
            return 'primal dual difference'
        else:
            logging.warning('index out of range [0,5]: {}'.format(idx))
            return ''

    overlap = [no_improvement_instances, always_improvement_instances, optimal_instances,
               incorrect_cut_amount_instances, numerical_issues_instances, primal_dual_instances]

    for i in range(6):
        for j in range(i + 1, 6):
            intersection = overlap[i].intersection(overlap[j])
            if len(intersection) > 0:
                print('Overlap of {} and {} has {} many instances'.format(instance_set_name(i), instance_set_name(j),
                                                                          len(intersection)), flush=True)

    # Dump the yml file containing all of our solve info into the right place. Use .YAML instead
    yaml_file = os.path.join(args.results_dir, 'grid_search.yaml')
    with open(yaml_file, 'w') as ss:
        yaml.dump(improvements, ss)
    print('Average Improvement: {}'.format(np.mean([improvements[i]['improvement'] for i in valid_instances])))
    # Dump the yml file containing solve info over all parameters in the sweep (not just the best)
    yaml_file = os.path.join(args.results_dir, 'all_grid_runs.yaml')
    with open(yaml_file, 'w') as ss:
        yaml.dump(full_results_dict, ss)
