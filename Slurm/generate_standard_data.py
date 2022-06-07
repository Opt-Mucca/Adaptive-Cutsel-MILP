#! /usr/bin/env python
import os
import argparse
import yaml
import logging
import numpy as np
import time
from utilities import run_python_slurm_job, get_filename, remove_temp_files, get_slurm_output_file, is_dir, \
    remove_slurm_files, is_file
import parameters


def run_slurm_job_with_random_seed(temp_dir, outfile_dir, instance_path, instance, rand_seed,
                                   time_limit, root, print_sol, print_stats, exclusive=True):
    """
    Calls a slurm job for solving a SCIP instance with the appropriate random seed
    Args:
        temp_dir: The temporary directory where all intermediate files get stored for the run (e.g. npy files)
        outfile_dir: The directory which we dump slurm out-files into
        instance_path: The path to the instance .mps
        instance: The instance base name for the problem
        rand_seed: The random seed which will be used to randomly initialise SCIP plugins
        time_limit: The time limit of the solve that we are going to run in our slurm job
        root: Whether the SCIP run in the job we're going to call is restricted to the root
        print_sol: Whether the .sol file produced by the run should be saved to temp_dir
        print_stats: Whether the statistics of the run should be output to a .stats files
        exclusive: Whether the run should be run on a node without interference (for reproducible timing)

    Returns:
        Nothing. It initialises all slurm jobs needed for standard cut-selector param solve info
    """

    # Call the slurm job
    ji = run_python_slurm_job(python_file='Slurm/solve_instance_seed_noise.py',
                              job_name='{}--{}'.format(instance, rand_seed),
                              outfile=os.path.join(outfile_dir, '%j__{}__seed__{}.out'.format(instance, rand_seed)),
                              time_limit=int(time_limit / 60) + 2,
                              arg_list=[temp_dir, instance_path,
                                        instance, rand_seed, 0, time_limit, root, print_sol, print_stats],
                              exclusive=exclusive)

    return ji


def run_clean_up_slurm_job_and_wait(outfile_dir, temp_dir, dependencies):
    """
    Function that submits a job with dependencies of all previous submitted jobs. This job then simply writes
    a cleaner.txt file when all those are done which signals that all other jobs are complete. We wait on the existence
    of that file
    Args:
        outfile_dir (dir): The directory where the outfile of this job will be dumped (outfile will be empty)
        temp_dir (dir): The directory where the txt file will be written
        dependencies (list): The list of dependencies for our job
    Returns:

    """

    _ = run_python_slurm_job(python_file='Slurm/safety_check.py',
                             job_name='cleaner',
                             outfile=os.path.join(outfile_dir, 'a--a--a.out'),
                             time_limit=10,
                             arg_list=[os.path.join(temp_dir, 'cleaner.txt')],
                             dependencies=dependencies)

    # Put the program to sleep until all of slurm jobs are complete
    time.sleep(30)
    while 'cleaner.txt' not in os.listdir(temp_dir):
        time.sleep(30)


def run_slurm_jobs_get_yml_and_log_files(data_dir, temp_dir, outfile_dir, instances, rand_seeds, root=True):
    """
    It solves a run for all instance and random seed combinations. It then creates a YML file on the statistics,
    a .stats file containing the SCIP output of the statistics, and a .log file containing the log output of the run.
    It does this for all combinations, and then filters out instances who failed on at-least one random seed,
    outputting the reason they failed.
    Args:
        data_dir (dir): Directory containing the mps files, where we will dump all final files
        temp_dir (dir): Directory where we dump all temporary files
        outfile_dir (dir): Directory where we dump all slurm .out files. These can later become .log files
        instances (list): List of instances we're interested in
        rand_seeds (list): List of random seeds we're interested in
        root (bool): Whether we restrict our solves to the root node or not

    Returns: The list of instances which solved over all random seeds successfully. Creates files of all relevant info

    """
    assert type(root) == bool

    # Set the time limit for the run. This simply depends on if the root node or not
    time_limit = parameters.ROOT_SOLVE_TIME_LIMIT if root else parameters.FULL_SOLVE_TIME_LIMIT

    # Change the outfile directory for this set of runs
    outfile_sub_dir_name = 'root_solve' if root else 'full_solve'
    outfile_dir = os.path.join(outfile_dir, outfile_sub_dir_name)
    assert not os.path.isdir(outfile_dir)
    os.mkdir(outfile_dir)

    # Create structures that will store reasons why the jobs failed
    invalid_sol_instances, invalid_mem_instances = set(), set()
    invalid_optimal_instances, invalid_time_instances = set(), set()
    invalid_num_cut_instances = set()

    # Get the list of slurm jobs we're going to run so we can run one final clean up job with dependencies
    slurm_job_ids = []

    # Start all the individual jobs that solve and instance and a random seed
    for instance in instances:
        for rand_seed in rand_seeds:
            save_default_cut_selector_param_npy_file(temp_dir, instance, rand_seed)
            mps_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
            ji = run_slurm_job_with_random_seed(temp_dir, outfile_dir, mps_file, instance, rand_seed,
                                                time_limit, root, False, True, exclusive=True)
            slurm_job_ids.append(ji)

    # Create the cleaner job that signals when all previously issued jobs are complete and wait on it.
    run_clean_up_slurm_job_and_wait(outfile_dir, temp_dir, slurm_job_ids)

    # Get all instances where the solve failed, or in the case of root node solves, some filtering criteria was hit
    problematic_instances = set()
    for instance in instances:
        for rand_seed in rand_seeds:
            yml_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='yml')
            stats_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='stats')
            if not os.path.isfile(yml_file) or not os.path.isfile(stats_file):
                logging.warning('Instance {} seed {} root {} run failed'.format(instance, rand_seed, root))
                problematic_instances.add(instance)
                outfile = get_slurm_output_file(outfile_dir, instance, rand_seed)
                invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances, invalid_time_instances = \
                    get_invalid_reason_from_out_file(outfile, invalid_sol_instances, invalid_mem_instances,
                                                     invalid_optimal_instances, invalid_time_instances, root=root)
                continue
            if root:
                with open(yml_file, 'r') as s:
                    yml_data = yaml.safe_load(s)
                if yml_data['status'] == 'timelimit':
                    invalid_time_instances.add(instance)
                    problematic_instances.add(instance)
                elif yml_data['status'] == 'optimal':
                    invalid_optimal_instances.add(instance)
                    problematic_instances.add(instance)
                else:
                    assert yml_data['status'] == 'nodelimit', '{} seed {} root status {}'.format(instance, rand_seed,
                                                                                                 yml_data['status'])
                    min_cuts = parameters.MIN_NUM_CUT_RATIO * parameters.NUM_CUT_ROUNDS * parameters.NUM_CUTS_PER_ROUND
                    if yml_data['num_cuts'] < min_cuts:
                        logging.warning('Instance {} seed {} is filtered'.format(instance, rand_seed))
                        invalid_num_cut_instances.add(instance)
                        problematic_instances.add(instance)

    print_invalid_instance_reasons(invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances,
                                   invalid_time_instances, invalid_num_cut_instances, use_cut_filtering=root)

    # Delete all files associated with the problematic instances
    for instance in list(problematic_instances):
        for rand_seed in rand_seeds:
            sol_path = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
            mps_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
            assert os.path.isfile(sol_path) and os.path.isfile(mps_file)
            os.remove(sol_path)
            os.remove(mps_file)
            if not root:
                r_log_file = get_filename(data_dir, instance, rand_seed, trans=True, root=True, sample_i=0, ext='log')
                r_yml_file = get_filename(data_dir, instance, rand_seed, trans=True, root=True, sample_i=0, ext='yml')
                r_st_file = get_filename(data_dir, instance, rand_seed, trans=True, root=True, sample_i=0, ext='stats')
                for file_to_remove in [r_log_file, r_yml_file, r_st_file]:
                    assert os.path.isfile(file_to_remove)
                    os.remove(file_to_remove)
            # Remove the stats file for the problematic run. It may not exist if the run failed
            stats_path = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='stats')
            if os.path.isfile(stats_path):
                os.remove(stats_path)

    instances = list(set(instances) - problematic_instances)

    # Now move the files we created for the non-problematic instances
    for instance in instances:
        for rand_seed in rand_seeds:
            # First do the YAML file
            yml_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='yml')
            new_yml_file = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='yml')
            assert os.path.isfile(yml_file) and not os.path.isfile(new_yml_file)
            os.rename(yml_file, new_yml_file)

            # Now do the log file
            out_file = get_slurm_output_file(outfile_dir, instance, rand_seed)
            new_out_file = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='log')
            assert os.path.isfile(out_file) and not os.path.isfile(new_out_file)
            os.rename(out_file, new_out_file)

            # Now do the stats file
            stats_path = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=0, ext='stats')
            new_path = get_filename(data_dir, instance, rand_seed, trans=True, root=root, sample_i=None, ext='stats')
            assert os.path.isfile(stats_path) and not os.path.isfile(new_path)
            os.rename(stats_path, new_path)

    return instances


def run_slurm_jobs_get_solution_files(data_dir, temp_dir, outfile_dir, instances, rand_seeds):
    """
    The function for generating calls to Slurm/solve_instance_seed_noise.py that will solve a SCIP
    instance that is only restricted by run-time. This run will be used to see if a feasible solution for the
    instance can be found. All instances that cannot find feasible instances within some time limit are then discarded.
    Args:
        data_dir (dir): Directory containing our pre-solved mps instances where we now will generate .sol files
        temp_dir (dir): Directory containing this function specific files
        outfile_dir (dir): Directory where our slurm log files will be output to
        instances (list): The list of instances we are interested in
        rand_seeds (list): The list of random seeds which we are interested in

    Returns:
        Produces the .sol files that we use as reference primal solutions for all later experiments, and returns a
        list of reduced instances, removing those instances which didn't produce any solution in the time limit
    """

    # Change the outfile directory for this set of runs
    outfile_dir = os.path.join(outfile_dir, 'get_sol_files')
    assert not os.path.isdir(outfile_dir)
    os.mkdir(outfile_dir)

    # Get the list of slurm jobs we're going to run so we can run one final clean up job with dependencies
    slurm_job_ids = []

    for instance in instances:
        for rand_seed in rand_seeds:
            save_default_cut_selector_param_npy_file(temp_dir, instance, rand_seed)
            mps_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')

            ji = run_slurm_job_with_random_seed(temp_dir, outfile_dir, mps_file, instance, rand_seed,
                                                parameters.SOL_FIND_TIME_LIMIT, False, True, False, exclusive=True)
            slurm_job_ids.append(ji)

    run_clean_up_slurm_job_and_wait(outfile_dir, temp_dir, slurm_job_ids)

    # Get all instances for which at-least one random seed failed to produce a solution
    problematic_instances = set()
    for instance in instances:
        for rand_seed in rand_seeds:
            sol_path = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
            if not os.path.isfile(sol_path):
                problematic_instances.add(instance)
                logging.warning('Instance {} with seed {} failed to produce a .sol file'.format(instance, rand_seed))

    # Remove the instance files for all the problematic instances
    for instance in list(problematic_instances):
        for rand_seed in rand_seeds:
            mps_file = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
            assert os.path.isfile(mps_file)
            os.remove(mps_file)

    # Update the instance list by removing the problematic instances
    instances = list(set(instances) - problematic_instances)

    # Now move the .sol files from the instances that worked into the directory containing the pre-solved mps files
    for instance in instances:
        for rand_seed in rand_seeds:
            sol_path = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
            new_sol_path = get_filename(data_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
            assert os.path.isfile(sol_path)
            assert not os.path.isfile(new_sol_path)
            os.rename(sol_path, new_sol_path)

    print('Filtered {} instances for not producing a .sol file in time'.format(len(problematic_instances)), flush=True)

    return instances


def save_default_cut_selector_param_npy_file(temp_dir, instance, rand_seed):
    """
    Creates a npy file for the default cut-selector parameter values
    Args:
        temp_dir (dir): Directory where we will dump the npy file
        instance (str): The name of the instance
        rand_seed (int): The random seed of the solve

    Returns: Nothing, just creates a file
    """

    cut_selector_params = np.array([0.0, 1.0, 0.1, 0.1])
    file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=0, ext='npy')
    np.save(file_name, cut_selector_params)

    return


def get_invalid_reason_from_out_file(outfile, invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances,
                                     invalid_time_instances, root=True, ignore_time_limit=False):
    """
    Gets the reason that a run was invalid. A run could be invalid for the following reasons:
    1. Solution not accepted
    2. Slurm memory limit has been hit
    3. The root solve was optimal
    4. time limit of the run was hit

    Args:
        outfile (is_file): The .out file produced by the run
        invalid_sol_instances (set): Set containing instances that did not accept pre-loaded solutions
        invalid_mem_instances (set): Set containing instances that exceeded memory limits
        invalid_optimal_instances (set): Set containing instances that solved to optimality at the root node
        invalid_time_instances (set): Instances that hit their time limits
        root (bool): Whether the run only was concerned with the root node
        ignore_time_limit: Whether the time limit should be ignored as a failure reason

    Returns:
        Updated appropriate lists with the invalid instances of the correct type
    """
    assert is_file(outfile) and outfile.endswith('.out'), print('{} is not an outfile'.format(outfile))

    instance = outfile.split('__')[1]

    with open(outfile, 'r') as f:
        lines = f.readlines()

    found_reason = False
    for line in lines:
        if 'Invalid input line' in line and 'solution file' in line:
            invalid_sol_instances.add(instance)
            found_reason = True
            break
        elif 'all 1 solutions given by solution candidate storage are infeasible' in line:
            invalid_sol_instances.add(instance)
            found_reason = True
            break
        elif 'error: Exceeded job memory limit' in line:
            invalid_mem_instances.add(instance)
            found_reason = True
            break
        elif 'error: Job' in line and 'exceeded memory limit' in line:
            invalid_mem_instances.add(instance)
            found_reason = True
            break
        elif 'problem is solved [optimal solution found]' in line and root:
            invalid_optimal_instances.add(instance)
            found_reason = True
            break
        elif 'CANCELLED' in line and 'DUE TO TIME LIMIT' in line:
            invalid_time_instances.add(instance)
            found_reason = True
            break
        elif 'solving was interrupted [time limit reached]' in line and not ignore_time_limit:
            invalid_time_instances.add(instance)
            found_reason = True
            break

    if not found_reason:
        print('Outfile {} was flagged as a failure, but no reason found'.format(outfile))
        quit()

    return invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances, invalid_time_instances


def print_invalid_instance_reasons(invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances,
                                   invalid_time_instances, invalid_cut_instances, use_cut_filtering=False):
    """

    Args:
        invalid_sol_instances (set): Set containing instances that did not accept pre-loaded solutions
        invalid_mem_instances (set): Set containing instances that exceeded memory limits
        invalid_optimal_instances (set): Set containing instances that solved to optimality at the root node
        invalid_time_instances (set): Instances that hit their time limits
        invalid_cut_instances (set): Instances that contained too little cuts
        use_cut_filtering (bool): Whether invalid cut instances has been populated

    Returns: Nothing, just prints information about the lists
    """
    assert type(invalid_sol_instances) == set
    assert type(invalid_mem_instances) == set
    assert type(invalid_optimal_instances) == set
    assert type(invalid_time_instances) == set
    assert type(invalid_cut_instances) == set

    print('Invalid Solution: {} many instances. {}'.format(len(invalid_sol_instances), invalid_sol_instances),
          flush=True)
    print('Too much memory: {} many instances. {}'.format(len(invalid_mem_instances), invalid_mem_instances),
          flush=True)
    print('Root optimal. {} many instances. {}'.format(len(invalid_optimal_instances), invalid_optimal_instances),
          flush=True)
    print('Too much time. {} many instances. {}'.format(len(invalid_time_instances), invalid_time_instances),
          flush=True)
    if use_cut_filtering:
        print('Too few cuts. {} many instances. {}'.format(len(invalid_cut_instances), invalid_cut_instances),
              flush=True)

    overlap = [invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances, invalid_time_instances,
               invalid_cut_instances]

    def instance_set_name(idx):
        if idx == 0:
            return 'invalid_sol_instances'
        elif idx == 1:
            return 'invalid_mem_instances'
        elif idx == 2:
            return 'invalid_optimal_instances'
        elif idx == 3:
            return 'invalid_time_instances'
        elif idx == 4:
            return 'invalid_cut_instances'
        else:
            logging.warning('index out of range [0,4]: {}'.format(idx))
            return ''

    for i in range(5):
        for j in range(i+1, 5):
            intersection = overlap[i].intersection(overlap[j])
            if len(intersection) > 0:
                print('Overlap of {} and {} has {} many instances'.format(instance_set_name(i), instance_set_name(j),
                                                                          len(intersection)), flush=True)
                for instance in list(intersection):
                    print('Instance {} had multiple unique fail reasons over its seeds. Reasons: {}, {}'.format(
                        instance, instance_set_name(i), instance_set_name(j)), flush=True)

    return


def pre_solve_instances(instances, instance_paths, sol_paths, rand_seeds, transformed_problem_dir, outfile_dir,
                        temp_dir, use_miplib_sols):
    """
    Function for pre-solving instances and filtering out those which take too much memory or took too long
    Args:
        instances (list): List of instances names
        instance_paths (list): List of instance paths (indices match instances)
        sol_paths (list): List of solution file paths (indices match instances)
        rand_seeds (list): List containing the random seeds we'll use
        transformed_problem_dir (dir): Directory where we will throw our pre-solved mps instance file
        outfile_dir (dir): The directory where we throw our slurm files
        temp_dir (dir): The directory where we throw our temporary files used just for this function
        use_miplib_sols (bool): Whether we are going to use the MIPLIB sols or not.

    Returns:
        Produces the pre-solved mps instances and returns a reduced list of instances
    """

    outfile_dir = os.path.join(outfile_dir, 'pre_solve')
    assert not os.path.isdir(outfile_dir)
    os.mkdir(outfile_dir)

    # Create structures that will store reasons why the jobs failed
    invalid_sol_instances, invalid_mem_instances = set(), set()
    invalid_optimal_instances, invalid_time_instances = set(), set()

    # Get the list of slurm jobs we submit so we can later make a final job with those dependencies
    slurm_job_ids = []

    for i, instance in enumerate(instances):
        for rand_seed in rand_seeds:
            outfile = os.path.join(outfile_dir, '%j__{}__seed__{}__pre-solve.out'.format(instance, rand_seed))
            ji = run_python_slurm_job(python_file='Slurm/presolve_instance.py',
                                      job_name='pre-solve--{}--{}'.format(instance, rand_seed),
                                      outfile=outfile, time_limit=int(parameters.PRESOLVE_TIME_LIMIT / 60) + 2,
                                      arg_list=[transformed_problem_dir, instance_paths[i], sol_paths[i], instance,
                                                rand_seed, use_miplib_sols],
                                      exclusive=True)
            slurm_job_ids.append(ji)

    run_clean_up_slurm_job_and_wait(outfile_dir, temp_dir, slurm_job_ids)

    # Get all instances that did not pre-solve for all random seeds
    invalid_instances = []
    for instance in instances:
        valid_instance = True
        for rand_seed in rand_seeds:
            mps_path = get_filename(transformed_problem_dir, instance, rand_seed, trans=True, root=False,
                                    sample_i=None, ext='mps')
            sol_path = get_filename(transformed_problem_dir, instance, rand_seed, trans=True, root=False,
                                    sample_i=None, ext='sol')
            if not os.path.isfile(mps_path) or (not os.path.isfile(sol_path) and parameters.USE_MIPLIB_SOLUTIONS):
                # If the MPS or SOL file does not exist, get the reason why the run failed and flag as failed
                valid_instance = False
                outfile = get_slurm_output_file(outfile_dir, instance, rand_seed)
                # Get reason why the run failed for this instance and random seed
                invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances, invalid_time_instances = \
                    get_invalid_reason_from_out_file(outfile, invalid_sol_instances, invalid_mem_instances,
                                                     invalid_optimal_instances, invalid_time_instances)
                logging.warning('Instance {} with random seed {} failed in pre-solve'.format(instance, rand_seed))
                break
        if not valid_instance:
            invalid_instances.append(instance)

    logging.warning('{} many instances {} failed pre-solve'.format(len(invalid_instances), invalid_instances))

    # Remove all files associated with these invalid instances. They might exist as only some seeds might fail.
    for instance in invalid_instances:
        for rand_seed in rand_seeds:
            mps_path = get_filename(transformed_problem_dir, instance, rand_seed, trans=True, root=False,
                                    sample_i=None, ext='mps')
            sol_path = get_filename(transformed_problem_dir, instance, rand_seed, trans=True, root=False,
                                    sample_i=None, ext='sol')
            for file_path in [mps_path, sol_path]:
                if os.path.isfile(file_path):
                    os.remove(file_path)

    print_invalid_instance_reasons(invalid_sol_instances, invalid_mem_instances, invalid_optimal_instances,
                                   invalid_time_instances, set(), use_cut_filtering=False)

    return list(set(instances) - set(invalid_instances))


def remove_previous_run_data(data_dir):
    """
    Function for removing previously calculated files.
    Args:
        data_dir: Directory containing our .mps files where we will dump our .yml and .sol files
    Returns:
        Nothing. Removes files from previous runs
    """

    # Cycle through the files
    for d_file in os.listdir(data_dir):
        for file_ext in ['.sol', '.yml', '.log', '.mps', '.npy', '.stats']:
            if d_file.endswith(file_ext):
                os.remove(os.path.join(data_dir, d_file))
                break

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_dir', type=is_dir)
    parser.add_argument('solution_dir', type=is_dir)
    parser.add_argument('transformed_problem_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('num_rand_seeds', type=int)
    args = parser.parse_args()

    # Remove all solve information from previous runs
    remove_previous_run_data(args.transformed_problem_dir)
    remove_temp_files(args.temp_dir)
    remove_slurm_files(args.outfile_dir)

    # Initialise a list of instances
    instance_names = []
    instance_file_paths = []
    sol_file_paths = []

    sol_files = os.listdir(args.solution_dir)
    for file in os.listdir(args.problem_dir):
        # Extract the instance
        assert file.endswith('.mps.gz'), 'File {} does not end with .mps.gz'.format(file)
        instance_name = file.split('.')[0]
        instance_names.append(instance_name)
        instance_file_paths.append(os.path.join(args.problem_dir, file))
        sol_file = instance_name + '.sol.gz'
        assert sol_file in sol_files, 'sol_file {} not found'.format(sol_file)
        sol_file_paths.append(os.path.join(args.solution_dir, sol_file))

    # Initialise the random seeds
    random_seeds = [random_seed for random_seed in range(1, args.num_rand_seeds + 1)]

    # First we pre-solve the instances and filter those which take too long or take too much memory
    print('Pre-Solving instances', flush=True)
    instance_names = pre_solve_instances(instance_names, instance_file_paths, sol_file_paths, random_seeds,
                                         args.transformed_problem_dir, args.outfile_dir, args.temp_dir,
                                         parameters.USE_MIPLIB_SOLUTIONS)
    remove_temp_files(args.temp_dir)

    if not parameters.USE_MIPLIB_SOLUTIONS:
        # We then filter those instances which cannot produce primal solutions
        print('Finding primal solutions to pre-solved instances', flush=True)
        instance_names = run_slurm_jobs_get_solution_files(args.transformed_problem_dir, args.temp_dir,
                                                           args.outfile_dir, instance_names, random_seeds)
        remove_temp_files(args.temp_dir)

    # We now produce YML files containing solve information for our root-node restricted solves.
    print('Producing root-node restricted solve statistics in YML files', flush=True)
    instance_names = run_slurm_jobs_get_yml_and_log_files(args.transformed_problem_dir, args.temp_dir, args.outfile_dir,
                                                          instance_names, random_seeds, True)
    remove_temp_files(args.temp_dir)

    # Finally we produce YML files containing solve information for our un-restricted solves
    if False:
        print('Producing unrestricted solve statistics in YML files', flush=True)
        instance_names = run_slurm_jobs_get_yml_and_log_files(args.transformed_problem_dir, args.temp_dir,
                                                              args.outfile_dir, instance_names, random_seeds, False)

    print('Finished!', flush=True)
