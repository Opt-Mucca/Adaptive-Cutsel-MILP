#! /usr/bin/env python
import os
import argparse
import yaml
from utilities import build_scip_model, str_to_bool, read_cut_selector_param_file, get_filename, is_dir, is_file


def run_instance(temp_dir, instance_path, instance, rand_seed, sample_i, time_limit, root, print_sol, print_stats):
    """
    The call to solve a single instance, where these runs will be done on slurm. The function loads the correct
    cut-selector parameters and then solves the appropriate SCIP instance.
    Args:
        temp_dir: The directory in which all temporary files per batch will be dumped then deleted (e.g. cut-sel params)
        instance_path: The path to the MIP .mps instance
        instance: The instance base name of the MIP file
        rand_seed: The random seed which will be used to shift all SCIP randomisation
        sample_i: The sample index so we can load the sampled cut-sel param YAML file
        time_limit: The time limit, if it exists for our SCIP instance. Negative time_limit means None
        root: A boolean for whether we should restrict our solve to the root node or not
        print_sol: Whether the .sol file from the run should be printed or not
        print_stats: Whether the .stats file from the run should be printed or not

    Returns:
        Nothing. All results from this run should be output to a file in temp_dir.
        The results should contain all information about the run, (e.g. cut-sel params, solve_time, dual_bound etc)
    """

    # Load the cut-selector params
    dir_cut_off, efficacy, int_support, obj_parallelism = read_cut_selector_param_file(temp_dir, instance, rand_seed,
                                                                                       sample_i)

    # Print out the cut-sel param values to the slurm .out file
    print('DIR: {}, EFF: {}, INT: {}, OBJ: {}'.format(dir_cut_off, efficacy, int_support, obj_parallelism), flush=True)

    # Build the initial SCIP model for the instance
    time_limit = None if time_limit < 0 else time_limit
    node_lim = 1 if root else -1
    propagation = False if root else True
    heuristics = False if root else True
    aggressive = True if root else False
    dummy_branch = True if root else False

    # Check is a solution file exists. This solution file should be next to the instance file
    if os.path.isfile(os.path.splitext(instance_path)[0] + '.sol'):
        sol_file = os.path.splitext(instance_path)[0] + '.sol'
    else:
        sol_file = None

    # Build the actual SCIP model from the information now
    scip = build_scip_model(instance_path, node_lim, rand_seed, False, propagation, True, heuristics, aggressive,
                            dummy_branch, time_limit=time_limit, sol_path=sol_file,
                            dir_cut_off=dir_cut_off, efficacy=efficacy, int_support=int_support,
                            obj_parallelism=obj_parallelism)

    # Solve the SCIP model and extract all solve information
    solve_model_and_extract_solve_info(scip, dir_cut_off, efficacy, int_support, obj_parallelism, rand_seed, sample_i,
                                       instance, temp_dir, root=root, print_sol=print_sol, print_stats=print_stats)

    # Free the SCIP instance
    scip.freeProb()

    return


def solve_model_and_extract_solve_info(scip, dir_cut_off, efficacy, int_support, obj_parallelism, rand_seed, sample_i,
                                       instance, temp_dir, root=True, print_sol=False, print_stats=False):
    """
    Solves the given SCIP model and after solving creates a YAML file with all potentially interesting
    solve information. This information will later be read and used to update the neural_network parameters
    Args:
        scip: The PySCIPOpt model that we want to solve
        dir_cut_off: The coefficient for the directed cut-off distance
        efficacy: The coefficient for the efficacy
        int_support: The coefficient for the integer support
        obj_parallelism: The coefficient for the objective function parallelism (see also the cosine similarity)
        rand_seed: The random seed used in the scip parameter settings
        sample_i: The sample index used to locate the correct cut-sel param values used
        instance: The instance base name of our problem
        temp_dir: The temporary file directory where we place all files that are batch-specific (e.g. cut-sel params)
        root: A kwarg that informs if the solve is restricted to the root node. Used for naming the yml file
        print_sol: A kwarg that informs if the .sol file from the run should be saved to a file
        print_stats: A kwarg that informs if the .stats file from the run should be saved to a file

    Returns:

    """

    # Solve the MIP instance. All parameters should be pre-set
    scip.optimize()

    # Initialise the dictionary that will store our solve information
    data = {}

    # Get the solve_time
    data['solve_time'] = scip.getSolvingTime()
    # Get the number of cuts applied
    data['num_cuts'] = scip.getNCutsApplied()
    # Get the number of nodes in our branch and bound tree
    data['num_nodes'] = scip.getNNodes()
    # Get the best primal solution if available
    data['primal_bound'] = scip.getObjVal() if len(scip.getSols()) > 0 else 1e+20
    # Get the gap provided a primal solution exists
    data['gap'] = scip.getGap() if len(scip.getSols()) > 0 else 1e+20
    # Get the best dual bound
    data['dual_bound'] = scip.getDualbound()
    # Get the number of LP iterations
    data['num_lp_iterations'] = scip.getNLPIterations()
    # Get the status of the solve
    data['status'] = scip.getStatus()

    # Save the sol file if we've been asked to
    if len(scip.getSols()) > 0 and print_sol:
        sol = scip.getBestSol()
        sol_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='sol')
        scip.writeSol(sol, sol_file)

    # Get the percentage of integer variables with fractional values. This includes implicit integer variables
    scip_vars = scip.getVars()
    non_cont_vars = [var for var in scip_vars if var.vtype() != 'CONTINUOUS']
    assert len(non_cont_vars) > 0
    if root:
        cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(var.getLPSol()))]
    else:
        assert len(scip.getSols()) > 0
        scip_sol = scip.getBestSol()
        cont_valued_non_cont_vars = [var for var in non_cont_vars if not scip.isZero(scip.frac(scip_sol[var]))]
    data['solution_fractionality'] = len(cont_valued_non_cont_vars) / len(non_cont_vars)

    # Add the cut-selector parameters
    data['dir_cut_off'] = dir_cut_off
    data['efficacy'] = efficacy
    data['int_support'] = int_support
    data['obj_parallelism'] = obj_parallelism

    # Get the primal dual integral. This is not really needed for root solves, but might be important to have
    # It is only accessible through the solver statistics. TODO: Write a wrapper function for this
    stat_file = get_filename(temp_dir, instance, rand_seed, trans=True, root=root, sample_i=sample_i, ext='stats')
    scip.writeStatistics(stat_file)
    with open(stat_file) as s:
        stats = s.readlines()
    # TODO: Make this safer to access.
    data['primal_dual_integral'] = float(stats[-3].split(':')[1].split('     ')[1])
    if not print_stats:
        os.remove(stat_file)

    # Dump the yml file containing all of our solve info into the right place
    yml_file = get_filename(temp_dir, instance, rand_seed=rand_seed, trans=True, root=root, sample_i=sample_i,
                            ext='yml')
    with open(yml_file, 'w') as s:
        yaml.dump(data, s)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('instance_path', type=is_file)
    parser.add_argument('instance', type=str)
    parser.add_argument('rand_seed', type=int)
    parser.add_argument('sample_i', type=int)
    parser.add_argument('time_limit', type=int)
    parser.add_argument('root', type=str_to_bool)
    parser.add_argument('print_sol', type=str_to_bool)
    parser.add_argument('print_stats', type=str_to_bool)
    args = parser.parse_args()

    # The main function call to run a SCIP instance with cut-sel params
    run_instance(args.temp_dir, args.instance_path, args.instance,
                 args.rand_seed, args.sample_i, args.time_limit, args.root, args.print_sol, args.print_stats)
