#! /usr/bin/env python
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
import numpy as np
import os
import argparse
import pdb
import yaml
import time
from utilities import is_dir, get_filename, get_instances, get_random_seeds, run_python_slurm_job, \
    get_slurm_output_file, remove_slurm_files, remove_temp_files


def train_smac_model(train_instance_dir, test_instance_dir, solution_dir,
                     default_results_dir, results_dir, temp_dir, outfile_dir, num_epochs,
                     seed_init):
    """
    Main function for training a SMAC model. It allows SMAC num_epoch many calls, where each call tests the parameter
    value over the entire training instance set.
    Currently SMAC4BB is used for black-box optimisation. This was selected following the descriptions presented in
    the following release paper:

    @inproceedings {lindauer-arxiv21a,
    author = {Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and
    Difan Deng and Carolin Benjamins and Tim Ruhkopf and René Sass and Frank Hutter},
    title = {SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
    booktitle = {ArXiv: 2109.09831},
    year = {2021},
    url = {https://arxiv.org/abs/2109.09831}
    }

    Args:
        train_instance_dir (dir): The directory containing all presolved training instances
        test_instance_dir (dir): The directory containing all presolved test instances
        solution_dir (dir): The directory containing all solution files
        default_results_dir (dir): The directory containing all default run results
        results_dir (dir): The directory where we should dump all final concatenated result files
        temp_dir (dir): The directory where all our temporary run specific files go
        outfile_dir (dir): The directory where we will redirect all individual run logs
        num_epochs (int): The number of times you want SMAC to be called (An iteration sees all instances once)
        # TODO: num_epochs is actually an upperbound on the number of evluations. Does SMAC ever use less?
        seed_init (int): The random seed used for the run (We used 2022)

    Returns:
        Nothing. It creates results file
    """

    # Get the numpy random state from the given seed_init. Get the instances and random seeds too
    random_state = np.random.RandomState(seed_init)
    train_instances = get_instances(train_instance_dir)
    test_instances = get_instances(test_instance_dir)
    instances = train_instances + test_instances
    rand_seeds = get_random_seeds(train_instance_dir)

    # Initialise the standard solve data
    standard_solve_data = {instance: {rand_seed: None for rand_seed in rand_seeds} for instance in instances}
    for instance in instances:
        for rand_seed in rand_seeds:
            yml_file = get_filename(default_results_dir, instance, rand_seed, trans=True, root=True, ext='yml')
            assert os.path.isfile(yml_file)
            with open(yml_file, 'r') as s:
                standard_solve_data[instance][rand_seed] = yaml.safe_load(s)

    global_run_information = {'instance_dir': train_instance_dir,
                              'solution_dir': solution_dir,
                              'temp_dir': temp_dir,
                              'outfile_dir': outfile_dir,
                              'rand_seeds': rand_seeds,
                              'standard_solve_data': standard_solve_data}

    # Define your hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter('dcd', 0, 1, default_value=0.0,
                                                              meta=global_run_information))
    configspace.add_hyperparameter(UniformFloatHyperparameter('eff', 0, 1, default_value=1.0))
    configspace.add_hyperparameter(UniformFloatHyperparameter('isp', 0, 1, default_value=0.1))
    configspace.add_hyperparameter(UniformFloatHyperparameter('obp', 0, 1, default_value=0.1))

    # Provide meta data for the optimization
    scenario = Scenario({
        'run_obj': 'quality',  # Optimize quality (alternatively runtime)
        'runcount-limit': num_epochs,  # Max number of function evaluations (the more the better)
        'cs': configspace,
    })

    bb = SMAC4BB(scenario=scenario, tae_runner=main_smac_call, rng=random_state)
    best_found_config = bb.optimize()

    print(best_found_config, flush=True)

    cut_selector_params = {cut_sel_param: best_found_config[cut_sel_param] for cut_sel_param in best_found_config}
    numpy_cut_sel_params = np.array([cut_selector_params[cut_sel_param] for cut_sel_param in cut_selector_params])
    numpy_cut_sel_params /= sum(numpy_cut_sel_params)
    print('Raw values: {}'.format(cut_selector_params), flush=True)
    print('Scaled values: {}'.format(numpy_cut_sel_params), flush=True)

    data = {}
    for instance_dir, train_test_or_valid in [(train_instance_dir, 'train'), (test_instance_dir, 'test')]:
        score, subset_data = run_scip_instances(instance_dir, solution_dir, temp_dir, outfile_dir, rand_seeds,
                                                standard_solve_data, numpy_cut_sel_params)
        for instance in subset_data:
            data[instance] = subset_data[instance]
        del subset_data
        print('For set {}: Best SMAC parameters have an average improvement of {}'.format(train_test_or_valid,
                                                                                          -1 * score), flush=True)

    with open(os.path.join(results_dir, 'smac_runs.yaml'), 'w') as s:
        yaml.dump(data, s)


def main_smac_call(config):
    """
    The function that SMAC calls for a single evaluation
    Args:
        config (configspace): Contains the parameter values SMAC uses for this run. We hide all our
        other arguments in the meta descriptions of these parameters

    Returns:
        The score of the run (lower is better). The relative improved primal-dual-difference over default settings
    """

    # Extract information from the meta description
    instance_dir = config.configuration_space['dcd'].meta['instance_dir']
    solution_dir = config.configuration_space['dcd'].meta['solution_dir']
    temp_dir = config.configuration_space['dcd'].meta['temp_dir']
    outfile_dir = config.configuration_space['dcd'].meta['outfile_dir']
    rand_seeds = config.configuration_space['dcd'].meta['rand_seeds']
    standard_solve_data = config.configuration_space['dcd'].meta['standard_solve_data']

    # Create the numpy array from the cut selector parameter values
    cut_selector_params = np.array([config['dcd'], config['eff'], config['isp'], config['obp']])
    cut_selector_params /= sum(cut_selector_params)

    score, _ = run_scip_instances(instance_dir, solution_dir, temp_dir, outfile_dir, rand_seeds, standard_solve_data,
                                  cut_selector_params)

    return score


def run_scip_instances(instance_dir, solution_dir, temp_dir, outfile_dir, rand_seeds, standard_solve_data,
                       cut_selector_params):
    """
    This function runs all SCIP instances for each instance and random seed combination (for the given constant
    cut selector parameter values). It then returns a dictionary containing the relative improvement, and the score
    that SMAC wants to minimise
    Args:
        instance_dir (dir): Directory containing all instance files (presolved)
        solution_dir (dir): Directory containing all solution files
        temp_dir (dir): Directory where all temporary files for this set of runs will be stored
        outfile_dir (dir): Directory where all .out files for slurm will be redirected to
        rand_seeds (list): The list of random seeds used for this set of runs
        standard_solve_data (dict): Dictionary containing solve data under standard SCIP conditions
        cut_selector_params (np-array): Array of the constant cut-selector parameters that will be used

    Returns:
        The score (relative primal-dual-difference improvement). Dictionary containing this information per instance
    """

    # Update the outfile directory
    print('Running jobs!', flush=True)
    run_id = len(os.listdir(outfile_dir))
    outfile_dir = os.path.join(outfile_dir, str(run_id))
    assert not os.path.isdir(outfile_dir)
    os.mkdir(outfile_dir)

    # Remove the temp files produced by the previous run
    remove_temp_files(args.temp_dir)

    # Get the instance names from the data directory
    instances = get_instances(instance_dir)

    # Generate the jobs
    slurm_job_ids = []
    for instance in instances:
        for rand_seed in rand_seeds:
            for sample_i in [0]:
                file_name = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=sample_i,
                                         ext='npy')
                np.save(file_name, cut_selector_params)
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
                                                    instance, rand_seed, sample_i, -1, True, True, False, False])
                slurm_job_ids.append(ji)

    # Now wait on the generated jobs by generating a checker job with dependencies
    signal_file = 'cleaner.txt'
    _ = run_python_slurm_job(python_file='Slurm/safety_check.py',
                             job_name='cleaner',
                             outfile=os.path.join(outfile_dir, 'a--a--a.out'),
                             time_limit=10,
                             arg_list=[os.path.join(temp_dir, signal_file)],
                             dependencies=slurm_job_ids)

    # Put the program to sleep until all of slurm jobs are complete
    time.sleep(10)
    while signal_file not in os.listdir(temp_dir):
        time.sleep(10)

    # Initialise the data directory which we'll load all of our solve information into
    data = {instance: {rand_seed: None for rand_seed in rand_seeds} for instance in instances}
    scores = {instance: {} for instance in instances}

    # Check if there were problems with the instance during solving. All successful solves should have a YAML file
    for instance in instances:
        for rand_seed in rand_seeds:
            for sample_i in [0]:
                # Get the yml file name that would represent this run
                yml_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=True, sample_i=sample_i,
                                        ext='yml')
                if not os.path.isfile(yml_file):
                    slurm_output_file = get_slurm_output_file(outfile_dir, instance, rand_seed, sample_i=sample_i)
                    print('Instance {} with seed {} and sample {} failed. Please go explore why. Outfile {}'.format(
                        instance, rand_seed, sample_i, slurm_output_file), flush=True)
                    print('Existing program!', flush=True)
                    quit()
                with open(yml_file, 'r') as s:
                    data[instance][rand_seed] = yaml.safe_load(s)

    # Get the scores from the instance
    for instance in instances:
        improvement = 0
        for rand_seed in rand_seeds:
            sd = standard_solve_data[instance][rand_seed]['primal_dual_difference']
            bd = data[instance][rand_seed]['primal_dual_difference']
            improvement += (sd - bd) / (abs(sd) + 1e-8)
        scores[instance]['improvement'] = improvement / len(rand_seeds)
        scores[instance]['parameters'] = [[float(cut_selector_params[0]), float(cut_selector_params[1]),
                                           float(cut_selector_params[2]), float(cut_selector_params[3])]]

    score = 0
    for instance in instances:
        assert scores[instance]['improvement'] is not None, 'Instance {}'.format(instance)
        score += scores[instance]['improvement']
    score /= len(instances)

    return -1 * score, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_instance_dir', type=is_dir)
    parser.add_argument('test_instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=is_dir)
    parser.add_argument('default_results_dir', type=is_dir)
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('num_epochs', type=int)
    parser.add_argument('seed_init', type=int)
    args = parser.parse_args()

    # Remove all slurm .out files produced by previous runs
    args.outfile_dir = os.path.join(args.outfile_dir, 'smac_runs')
    if not os.path.isdir(args.outfile_dir):
        os.mkdir(args.outfile_dir)
    else:
        remove_slurm_files(args.outfile_dir)

    train_smac_model(args.train_instance_dir, args.test_instance_dir,
                     args.solution_dir, args.default_results_dir, args.results_dir, args.temp_dir,
                     args.outfile_dir, args.num_epochs, args.seed_init)
