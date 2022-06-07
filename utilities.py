#! /usr/bin/env python
import os
import numpy as np
import torch
import subprocess
import shutil
import logging
import argparse
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule, SCIP_PRESOLTIMING, SCIP_PROPTIMING
from ConstraintHandler.ConstraintHandler import RepeatSepaConshdlr
from CutSelectors.FixedAmountCutsel import FixedAmountCutsel
import parameters


def build_scip_model(instance_path, node_lim, rand_seed, pre_solve, propagation, separators, heuristics,
                     aggressive_sep, dummy_branch_rule, time_limit=None, sol_path=None,
                     dir_cut_off=0.0, efficacy=1.0, int_support=0.1, obj_parallelism=0.1):
    """
    General function to construct a PySCIPOpt model.

    Args:
        instance_path: The path to the instance
        node_lim: The node limit
        rand_seed: The random seed
        pre_solve: Whether pre-solve should be enabled or disabled
        propagation: Whether propagators should be enabled or disabled
        separators: Whether separators should be enabled or disabled
        heuristics: Whether heuristics should be enabled or disabled
        aggressive_sep: Whether we want aggressive separators. Disabling separators overrides this.
        dummy_branch_rule: This is to cover a 'feature' of SCIP where by default strong branching is done and this can
                           give information about nodes beneath the node limit. So we add a branch-rule that can't.
        time_limit: The time_limit of the model
        sol_path: An optional path to a valid .sol file containing a primal solution to the instance
        dir_cut_off: The directed cut off weight that is applied to the custom cut-selector
        efficacy: The efficacy weight that is applied to the custom cut-selector
        int_support: The integer support weight that is applied to the custom cut-selector
        obj_parallelism: The objective parallelism weight that is applied to the custom cut-selector

    Returns:
        pyscipopt model

    """
    assert os.path.exists(instance_path)
    assert type(node_lim) == int and type(rand_seed) == int
    assert all([type(param) == bool for param in [pre_solve, propagation, separators, heuristics, aggressive_sep]])

    scip = Model()
    scip.setParam('limits/nodes', node_lim)
    scip.setParam('randomization/randomseedshift', rand_seed)
    if not pre_solve:
        # Printing the transformed MPS files keeps the fixed variables and this drastically changes the solve
        # functionality after reading in the model and re-solving. So set one round of pre-solve to remove these
        # Additionally, we want constraints to be the appropriate types and not just linear for additional separators
        scip.setParam('presolving/maxrounds', 1)
        # scip.setPresolve(SCIP_PARAMSETTING.OFF)
    if not propagation:
        scip.disablePropagation()
    if not separators:
        scip.setSeparating(SCIP_PARAMSETTING.OFF)
    elif aggressive_sep:
        # Set the number of rounds we want and the number of cuts per round that will be forced
        num_rounds = parameters.NUM_CUT_ROUNDS
        cuts_per_round = parameters.NUM_CUTS_PER_ROUND
        # Create a dummy constraint handler that forces the num_rounds amount of separation rounds
        constraint_handler = RepeatSepaConshdlr(scip, num_rounds)
        scip.includeConshdlr(constraint_handler, "RepeatSepa", "Forces a certain number of separation rounds",
                             sepapriority=-1, enfopriority=1, chckpriority=-1, sepafreq=-1, propfreq=-1,
                             eagerfreq=-1, maxprerounds=-1, delaysepa=False, delayprop=False, needscons=False,
                             presoltiming=SCIP_PRESOLTIMING.FAST, proptiming=SCIP_PROPTIMING.AFTERLPNODE)
        # Create a cut-selector with highest priority that forces cuts_per_rounds to be selected each round
        cut_selector = FixedAmountCutsel(num_cuts_per_round=cuts_per_round, dir_cutoff_dist_weight=dir_cut_off,
                                         efficacy_weight=efficacy, int_support_weight=int_support,
                                         obj_parallel_weight=obj_parallelism)
        scip.includeCutsel(cut_selector, 'FixedAmountCutSel', 'Tries to add the same number of cuts per round',
                           1000000)
        # Set the separator parameters
        scip.setParam('separating/maxstallroundsroot', num_rounds)
        scip = set_scip_separator_params(scip, num_rounds, -1, cuts_per_round, cuts_per_round, 0)
    else:
        # scip = set_scip_separator_params(scip, -1, -1, 5000, 100, 10)
        scip = set_scip_cut_selector_params(scip, dir_cut_off, efficacy, int_support, obj_parallelism)
    if not heuristics:
        scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    if dummy_branch_rule:
        scip.setParam('branching/leastinf/priority', 10000000)
    if time_limit is not None:
        scip.setParam('limits/time', time_limit)

    # We do not want oribtope constraints as they're difficult to represent in the bipartite graph
    scip.setParam('misc/usesymmetry', 0)

    # read in the problem
    scip.readProblem(instance_path)

    if sol_path is not None:
        assert os.path.isfile(sol_path) and '.sol' in sol_path
        # Create the solution to add to SCIP
        sol = scip.readSolFile(sol_path)
        # Add the solution. This automatically frees the loaded solution
        scip.addSol(sol)

    return scip


def set_scip_cut_selector_params(scip, dir_cut_off, efficacy, int_support, obj_parallelism):
    """
    Sets the SCIP hybrid cut-selector parameter values in the weighted sum
    Args:
        scip: The PySCIPOpt model
        dir_cut_off: The coefficient of the directed cut-off distance
        efficacy: The coefficient of the efficacy
        int_support: The coefficient of the integer support
        obj_parallelism: The coefficient of the objective value parallelism (cosine similarity)

    Returns:
        The PySCIPOpt model with set parameters
    """
    scip.setParam("cutselection/hybrid/dircutoffdistweight", max(dir_cut_off, 0))
    scip.setParam("cutselection/hybrid/efficacyweight", max(efficacy, 0))
    scip.setParam("cutselection/hybrid/intsupportweight", max(int_support, 0))
    scip.setParam("cutselection/hybrid/objparalweight", max(obj_parallelism, 0))

    return scip


def set_scip_separator_params(scip, max_rounds_root=-1, max_rounds=-1, max_cuts_root=10000, max_cuts=10000,
                              frequency=10):
    """
    Function for setting the separator params in SCIP. It goes through all separators, enables them at all points
    in the solving process,
    Args:
        scip: The SCIP Model object
        max_rounds_root: The max number of separation rounds that can be performed at the root node
        max_rounds: The max number of separation rounds that can be performed at any non-root node
        max_cuts_root: The max number of cuts that can be added per round in the root node
        max_cuts: The max number of cuts that can be added per node at any non-root node
        frequency: The separators will be called each time the tree hits a new multiple of this depth
    Returns:
        The SCIP Model with all the appropriate parameters now set
    """

    assert type(max_cuts) == int and type(max_rounds) == int
    assert type(max_cuts_root) == int and type(max_rounds_root) == int

    # First for the aggregation heuristic separator
    scip.setParam('separating/aggregation/freq', frequency)
    scip.setParam('separating/aggregation/maxrounds', max_rounds)
    scip.setParam('separating/aggregation/maxroundsroot', max_rounds_root)
    scip.setParam('separating/aggregation/maxsepacuts', 10000)
    scip.setParam('separating/aggregation/maxsepacutsroot', 10000)

    # Now the Chvatal-Gomory w/ MIP separator
    # scip.setParam('separating/cgmip/freq', frequency)
    # scip.setParam('separating/cgmip/maxrounds', max_rounds)
    # scip.setParam('separating/cgmip/maxroundsroot', max_rounds_root)

    # The clique separator
    scip.setParam('separating/clique/freq', frequency)
    scip.setParam('separating/clique/maxsepacuts', 10000)

    # The close-cuts separator
    scip.setParam('separating/closecuts/freq', frequency)

    # The CMIR separator
    scip.setParam('separating/cmir/freq', frequency)

    # The Convex Projection separator
    scip.setParam('separating/convexproj/freq', frequency)
    scip.setParam('separating/convexproj/maxdepth', -1)

    # The disjunctive cut separator
    scip.setParam('separating/disjunctive/freq', frequency)
    scip.setParam('separating/disjunctive/maxrounds', max_rounds)
    scip.setParam('separating/disjunctive/maxroundsroot', max_rounds_root)
    scip.setParam('separating/disjunctive/maxinvcuts', 10000)
    scip.setParam('separating/disjunctive/maxinvcutsroot', 10000)
    scip.setParam('separating/disjunctive/maxdepth', -1)

    # The separator for edge-concave function
    scip.setParam('separating/eccuts/freq', frequency)
    scip.setParam('separating/eccuts/maxrounds', max_rounds)
    scip.setParam('separating/eccuts/maxroundsroot', max_rounds_root)
    scip.setParam('separating/eccuts/maxsepacuts', 10000)
    scip.setParam('separating/eccuts/maxsepacutsroot', 10000)
    scip.setParam('separating/eccuts/maxdepth', -1)

    # The flow cover cut separator
    scip.setParam('separating/flowcover/freq', frequency)

    # The gauge separator
    scip.setParam('separating/gauge/freq', frequency)

    # Gomory MIR cuts
    scip.setParam('separating/gomory/freq', frequency)
    scip.setParam('separating/gomory/maxrounds', max_rounds)
    scip.setParam('separating/gomory/maxroundsroot', max_rounds_root)
    scip.setParam('separating/gomory/maxsepacuts', 10000)
    scip.setParam('separating/gomory/maxsepacutsroot', 10000)

    # The implied bounds separator
    scip.setParam('separating/impliedbounds/freq', frequency)

    # The integer objective value separator
    scip.setParam('separating/intobj/freq', frequency)

    # The knapsack cover separator
    scip.setParam('separating/knapsackcover/freq', frequency)

    # The multi-commodity-flow network cut separator
    scip.setParam('separating/mcf/freq', frequency)
    scip.setParam('separating/mcf/maxsepacuts', 10000)
    scip.setParam('separating/mcf/maxsepacutsroot', 10000)

    # The odd cycle separator
    scip.setParam('separating/oddcycle/freq', frequency)
    scip.setParam('separating/oddcycle/maxrounds', max_rounds)
    scip.setParam('separating/oddcycle/maxroundsroot', max_rounds_root)
    scip.setParam('separating/oddcycle/maxsepacuts', 10000)
    scip.setParam('separating/oddcycle/maxsepacutsroot', 10000)

    # The rapid learning separator
    scip.setParam('separating/rapidlearning/freq', frequency)

    # The strong CG separator
    scip.setParam('separating/strongcg/freq', frequency)

    # The zero-half separator
    scip.setParam('separating/zerohalf/freq', frequency)
    scip.setParam('separating/zerohalf/maxcutcands', 100000)
    scip.setParam('separating/zerohalf/maxrounds', max_rounds)
    scip.setParam('separating/zerohalf/maxroundsroot', max_rounds_root)
    scip.setParam('separating/zerohalf/maxsepacuts', 10000)
    scip.setParam('separating/zerohalf/maxsepacutsroot', 10000)

    # The rlt separator
    scip.setParam('separating/rlt/freq', frequency)
    scip.setParam('separating/rlt/maxncuts', 10000)
    scip.setParam('separating/rlt/maxrounds', max_rounds)
    scip.setParam('separating/rlt/maxroundsroot', max_rounds_root)

    # Now the general cut and round parameters
    scip.setParam("separating/maxroundsroot", max_rounds_root)
    scip.setParam("separating/maxstallroundsroot", max_rounds_root)
    scip.setParam("separating/maxcutsroot", max_cuts_root)

    scip.setParam("separating/maxrounds", max_rounds)
    scip.setParam("separating/maxstallrounds", 1)
    scip.setParam("separating/maxcuts", max_cuts)

    return scip


def get_covariance_matrix(epoch_i, num_epochs, start_val=0.01, end_val=0.001):
    """
    Function for getting the covariance matrix we use in our MultivariateNormal distribution to sample our
    cut-selector parameter values for a single run. The covariance matrix is diagonal, so we assume no interaction
    between the different parameters. The matrix looks like:
    |d  0  0  0|
    |0  d  0  0|
    |0  0  d  0|
    |0  0  0  d|
    where,
    d(0) = start_val, d(num_epochs) = end_val, and d(i) = d(0) - (i * (d(0) - d(num_epochs))) / num_epochs
    Args:
        epoch_i: The current epoch
        num_epochs: The total number of epochs in our experiment
        start_val: The diagonal of the covariance matrix that we begin with
        end_val: The diagonal of the covariance matrix that we end with

    Returns:
        The covariance matrix
    """

    assert end_val <= start_val, print('start covariance diag {}, end {}'.format(start_val, end_val))
    assert end_val >= 0 and start_val >= 0, print('start covariance diag {}, end {}'.format(start_val, end_val))
    assert epoch_i <= num_epochs
    assert epoch_i >= 0 and num_epochs >= 0

    d = start_val - ((epoch_i * (start_val - end_val)) / num_epochs)

    return d * torch.eye(4)


def read_feature_vector_files(problem_dir, instance, rand_seed, torch_output=False):
    """
    This function just grabs the pre-calculated bipartite graph features from generate_features that have been
    written to files.
    Args:
        problem_dir: The directory containing all appropriate files
        instance: The instance name
        rand_seed: The SCIP random seed shift used in the model pre-solving
        torch_output: Boolean on whether you want torch or numpy as the output format

    Returns:
        The edge_indices, coefficients, col_features, row_features of the bipartite graph representation of the instance
    """

    edge_indices = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__edge_indices.npy'.format(instance, rand_seed)))
    coefficients = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__coefficients.npy'.format(instance, rand_seed)))
    col_features = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__col_features.npy'.format(instance, rand_seed)))
    row_features = np.load(
        os.path.join(problem_dir, '{}__trans__seed__{}__row_features.npy'.format(instance, rand_seed)))

    if torch_output:
        # Transform the numpy arrays into the correct torch types
        edge_indices = torch.from_numpy(edge_indices).long()
        coefficients = torch.from_numpy(coefficients).to(dtype=torch.float32)
        col_features = torch.from_numpy(col_features).to(dtype=torch.float32)
        row_features = torch.from_numpy(row_features).to(dtype=torch.float32)

    return edge_indices, coefficients, col_features, row_features


def read_cut_selector_param_file(problem_dir, instance, rand_seed, sample_i):
    """
    This function just grabs the pre-calculated cut-selector parameters that have been written to file.
    Args:
        problem_dir: The directory containing all appropriate files
        instance: The instance name
        rand_seed: The SCIP random seed shift used in the model pre-solving
        sample_i: The sample index used in the run to produce the saved file

    Returns:
        dir_cut_off, efficacy, int_support, obj_parallelism
    """

    # Get the saved file
    file_name = get_filename(problem_dir, instance, rand_seed, trans=True, root=False, sample_i=sample_i, ext='npy')
    cut_selector_params = np.load(file_name)

    dir_cut_off, efficacy, int_support, obj_parallelism = cut_selector_params.tolist()

    return dir_cut_off, efficacy, int_support, obj_parallelism


def remove_slurm_files(outfile_dir):
    """
    Removes all files from outfile_dir.
    Args:
        outfile_dir: The output directory containing all of our slurm .out files

    Returns:
        Nothing. It simply deletes the files
    """

    assert not outfile_dir == '/' and not outfile_dir == ''

    # Delete everything
    shutil.rmtree(outfile_dir)

    # Make the directory itself again
    os.mkdir(outfile_dir)

    return


def remove_temp_files(temp_dir):
    """
    Removes all files from the given directory
    Args:
        temp_dir: The directory containing all information that is batch specific

    Returns:
        Nothing, the function deletes all files in the given directory
    """

    # Get all files in the directory
    files = os.listdir(temp_dir)

    # Now cycle through the files and delete them
    for file in files:
        os.remove(os.path.join(temp_dir, file))

    return


def remove_instance_solve_data(data_dir, instance, suppress_warnings=False):
    """
    Removes all .mps, .npy, .yml, .sol, and .log files associated with the instance.
    Args:
        data_dir: The directory where we store all of our instance data
        instance: The instance name
        suppress_warnings: Whether the warnings of the files being deletes should be suppressed
    Returns:
        Nothing
    """

    assert os.path.isdir(data_dir)
    assert type(instance) == str

    # Get all files in the directory
    files = os.listdir(data_dir)

    # Get all files that being with our instance
    files = [file for file in files if file.split('__')[0] == instance]

    for file in files:
        if file.endswith('.yml') or file.endswith('.log') or file.endswith('.sol') or file.endswith('.mps')\
                or file.endswith('.npy') or file.endswith('.stats'):
            if not suppress_warnings:
                logging.warning('Deleting file {}'.format(os.path.join(data_dir, file)))
            os.remove(os.path.join(data_dir, file))

    return


def run_python_slurm_job(python_file, job_name, outfile, time_limit, arg_list, dependencies=None, num_cpus=1,
                         exclusive=False):
    """
    Function for calling a python file through slurm. This offloads the job from the current call
    and let's multiple processes run simultaneously. These processes can then share information though input output.
    Note: Spawned processes cannot directly communicate with each other
    Args:
        python_file: The python file that wil be run
        job_name: The name to give the python run in slurm
        outfile: The file in which all output from the python run will be stored
        time_limit: The time limit on the slurm job in minutes
        arg_list: The list containing all args that will be added to the python call
        dependencies: A list of slurm job ID dependencies that must first complete before this job starts
        num_cpus: The number of CPUS assigned to the single job
        exclusive: Whether the job should be the only jbo to run on a node. Doing this ignores mem and num_cpus
    Returns:
        Nothing. It simply starts a python job through the command line that will be run in slurm
    """

    if dependencies is None:
        dependencies = []
    assert os.path.isfile(python_file) and python_file.endswith('.py')
    assert not os.path.isfile(outfile) and outfile.endswith('.out'), '{}'.format(outfile)
    assert os.path.isdir(os.path.dirname(outfile)), '{}'.format(outfile)
    assert type(time_limit) == int and 0 <= time_limit <= 1e+8
    assert type(arg_list) == list
    assert dependencies is None or (type(dependencies) == list and
                                    all(type(dependency) == int for dependency in dependencies))

    # Get the current working environment.
    my_env = os.environ.copy()

    # Give the base command line call for running a single slurm job through shell.
    cmd_1 = ['sbatch',
             '--job-name={}'.format(job_name),
             '--time=0-00:{}:00'.format(time_limit)]

    if exclusive:
        # This flag makes the timing reproducible, as no memory is shared between it and other jobs.
        cmd_2 = ['--exclusive']
    else:
        # We don't run exclusive always as we want more throughput. The run is still deterministic, but time can vary
        cmd_2 = ['--cpus-per-task={}'.format(num_cpus)]
        # If you wanted to add memory limits; '--mem={}'.format(mem), where mem is in MB, e.g. 8000=8GB
    if dependencies is not None and len(dependencies) > 0:
        # Add the dependencies if they exist
        dependency_str = ''.join([str(dependency) + ':' for dependency in dependencies])[:-1]
        cmd_2 += ['--dependency=afterany:{}'.format(dependency_str)]

    cmd_3 = ['-p',
             parameters.SLURM_QUEUE,
             '--output',
             outfile,
             '--error',
             outfile,
             '{}'.format(python_file)]

    cmd = cmd_1 + cmd_2 + cmd_3

    # Add all arguments of the python file afterwards
    for arg in arg_list:
        cmd.append('{}'.format(arg))

    # Run the command in shell.
    p = subprocess.Popen(cmd, env=my_env, stdout=subprocess.PIPE)
    p.wait()

    # Now access the stdout of the subprocess for the job ID
    job_line = ''
    for line in p.stdout:
        job_line = str(line.rstrip())
        break
    assert 'Submitted batch job' in job_line, print(job_line)
    job_id = int(job_line.split(' ')[-1].split("'")[0])

    del p

    return job_id


def get_filename(parent_dir, instance, rand_seed=None, trans=False, root=False, sample_i=None, ext='yml'):
    """
    The main function for retrieving the file names for all non-temporary files. It is a shortcut to avoid constantly
    rewriting the names of the different files, such as the .yml, .sol, .log and .mps files
    Args:
        parent_dir: The parent directory where the file belongs
        instance: The instance name of the SCIP problem
        rand_seed: The random seed used in the SCIP run
        trans: Whether the filename contains the substring trans (problem has been pre-solved)
        root: If root should be included in the file name
        sample_i: The sample index used to perturb the SCIP cut-sel params
        ext: The extension of the file, e.g. yml or sol
    Returns:
        The filename e.g. 'parent_dir/toll-like__trans__seed__2__sample__2.mps'
    """

    # Initialise the base_file name. This always contains the instance name
    base_file = instance
    if trans:
        base_file += '__trans'
    if root:
        base_file += '__root'
    if rand_seed is not None:
        base_file += '__seed__{}'.format(rand_seed)
    if not (sample_i is False or sample_i is None):
        base_file += '__sample__{}'.format(sample_i)

    # Add the extension to the base file
    if ext is not None:
        base_file += '.{}'.format(ext)

    # Now join the file with its parent dir
    return os.path.join(parent_dir, base_file)


def get_slurm_output_file(outfile_dir, instance, rand_seed):
    """
    Function for getting the slurm output log for the current run.
    Args:
        outfile_dir: The directory containing all slurm .log files
        instance: The instance name
        rand_seed: The instance random seed
    Returns:
        The slurm .out file which is currently being used
    """

    assert os.path.isdir(outfile_dir)
    assert type(instance) == str
    assert type(rand_seed) == int

    # Get all slurm out files
    out_files = os.listdir(outfile_dir)

    # Get a unique substring that will only be contained for a single run
    file_substring = '__{}__seed__{}'.format(instance, rand_seed)

    unique_file = [out_file for out_file in out_files if file_substring in out_file]
    assert len(unique_file) == 1, 'Instance {} with rand_seed {} has no outfile in {}'.format(instance, rand_seed,
                                                                                              outfile_dir)
    return os.path.join(outfile_dir, unique_file[0])


def str_to_bool(word):
    """
    This is used to check if a string is trying to represent a boolean True.
    We need this because argparse doesnt by default have such a function, and using using bool('False') evaluate to True
    Args:
        word: The string we want to convert to a boolean
    Returns:
        Whether the string is representing True or not.
    """
    assert type(word) == str
    return word.lower() in ["yes", "true", "t", "1"]


def is_dir(path):
    """
    This is used to check if a string is trying to represent a directory when we parse it into argparse.
    Args:
        path: The path to a directory
    Returns:
        The string path if it is a valid directory else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isdir(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid directory'.format(path))
    else:
        return path


def is_file(path):
    """
    This is used to check if a string is trying to represent a file when we parse it into argparse.
    Args:
        path: The path to a file
    Returns:
        The string path if it is a valid file else we raise an error
    """
    assert type(path) == str, print('{} is not a string!'.format(path))
    exists = os.path.isfile(path)
    if not exists:
        raise argparse.ArgumentTypeError('{} is not a valid file'.format(path))
    else:
        return path

