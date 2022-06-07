#! /usr/bin/env python
import os
import argparse
import numpy as np
from utilities import build_scip_model, is_dir, get_filename, remove_slurm_files, remove_temp_files
from BranchRules.RootNodeFeatureExtractorBranchRule import RootNodeFeatureExtractor


def generate_feature_vectors(instance, rand_seed, transformed_problem_dir):

    # Load the pre-solved instance and generate the feature vector
    file = get_filename(transformed_problem_dir, instance, rand_seed, trans=True, root=False, sample_i=None, ext='mps')
    assert os.path.isfile(file)
    scip = build_scip_model(file, -1, rand_seed, False, False, False, False, False, False)
    feature_extractor = RootNodeFeatureExtractor(scip)
    scip.includeBranchrule(feature_extractor, "feature_extractor", "extract features of LP solution at root node",
                           priority=10000000, maxdepth=-1, maxbounddist=1)
    scip.optimize()
    feature_dict = feature_extractor.features
    scip.freeProb()

    # Extract the features from the read friendly dictionary format
    edge_indices, coefficients, col_features, row_features = extract_features_from_dict(feature_dict)

    # Save the various instance features into different files.
    file = get_filename(transformed_problem_dir, instance, rand_seed=rand_seed, trans=True, root=False, sample_i=None,
                        ext='npy')
    file_base = os.path.splitext(file)[0]
    np.save('{}__edge_indices.npy'.format(file_base), edge_indices)
    np.save('{}__coefficients.npy'.format(file_base), coefficients)
    np.save('{}__row_features.npy'.format(file_base), row_features)
    np.save('{}__col_features.npy'.format(file_base), col_features)

    return


def ensure_features_are_correct(feature_dict):
    """
    A potentially unnecessary safety check. This is just to ensure that our features are as they should be.
    Args:
        feature_dict: A dictionary containing all feature information to encode a GNN data point

    Returns:
        Nothing, this is just to make sure that SCIP did not return anything surprising.
    """

    # Make some basic safety functions to ensure that we have the correct types.
    def is_an(val, normalised=False):
        # Numpy ints / floats shouldn't be used until this point
        return (type(val) == float or type(val) == int) and (not normalised or (-1 - 1e-5 <= val <= 1 + 1e-5))

    def is_bin(val):
        return val == 1 or val == 0

    # Now get the variable (column's of the LP relaxation) features
    assert 'cols' in feature_dict
    assert len(feature_dict['cols']) > 1 and all(type(col_i) == int for col_i in feature_dict['cols'].keys())
    # Now make sure that all features were extracted. SCIP should have produced warnings but it's easier to debug here
    for col_i in feature_dict['cols'].keys():
        col = feature_dict['cols'][col_i]
        assert 'sol_val' in col and is_an(col['sol_val']), 'col {}. sol_val key: {}, value {}'.format(
            col_i, 'sol_val' in col, col['sol_val'])
        for vtype in ['BINARY', 'INTEGER', 'CONTINUOUS', 'IMPLINT']:
            assert vtype in col and is_bin(col[vtype]), 'col {}. {} key: {}, value {}'.format(
                col_i, vtype, vtype in col, col[vtype])
        assert 'obj_coeff' in col and is_an(col['obj_coeff'], True), 'col {}. obj_coeff key {}, value {}'.format(
            col_i, 'obj_coeff' in col, col['obj_coeff'])
        """assert 'has_lb' in col and is_bin(col['has_lb']), 'col {}. has_lb key {}, value {}'.format(
            col_i, 'has_lb' in col, col['has_lb'])
        assert 'has_ub' in col and is_bin(col['has_ub']), 'col {}. has_ub key {}, value {}'.format(
            col_i, 'has_ub' in col, col['has_ub'])"""
        assert 'has_lb' in col and is_an(col['has_lb']), 'col {}. has_lb key {}, value {}'.format(
            col_i, 'has_lb' in col, col['has_lb'])
        assert 'has_ub' in col and is_an(col['has_ub']), 'col {}. has_ub key {}, value {}'.format(
            col_i, 'has_ub' in col, col['has_ub'])
        # TODO: The has_lb / has_ub key is misleading, and it's now a normalised bound value.
        for btype in ['lower', 'basic', 'upper', 'zero']:
            assert btype in col and is_bin(col[btype]), 'col {}. {} key: {}, value {}'.format(
                col_i, btype, btype in col, col[btype])
        assert 'red_cost' in col and is_an(col['red_cost']), 'col {}. red_cost key {}, value {}'.format(
            col_i, 'red_cost' in col, col['red_cost'])
        assert 'sol_frac' in col and is_an(col['sol_frac'], True), 'col {}. sol_frac key {}, value {}'.format(
            col_i, 'sol_frac' in col, col['sol_frac'])
        assert 'is_at_lb' in col and is_bin(col['is_at_lb']), 'col {}. is_at_lb key {}, value {}'.format(
            col_i, 'is_at_lb' in col, col['is_at_lb'])
        assert 'is_at_ub' in col and is_bin(col['is_at_ub']), 'col {}. is_at_ub key {}, value {}'.format(
            col_i, 'is_at_ub' in col, col['is_at_ub'])

    # Now we do this for the constraint features (rows in our LP root relaxation)
    assert 'rows' in feature_dict
    assert len(feature_dict['rows']) > 1 and all(type(row_i) == int for row_i in feature_dict['rows'].keys())
    for row_i in feature_dict['rows'].keys():
        row = feature_dict['rows'][row_i]
        assert 'obj_cosine' in row and is_an(row['obj_cosine'], True), 'row {}. obj_cosine key {}, value {}'.format(
            row_i, 'obj_cosine' in row, row['obj_cosine'])
        assert 'bias' in row and is_an(row['bias']), 'row {}. bias key {}, value {}'.format(
            row_i, 'bias' in row, row['bias'])
        assert 'is_at_lb' in row and is_bin(row['is_at_lb']), 'row {}. is_at_lb key {}, value {}'.format(
            row_i, 'is_at_lb' in row, row['is_at_lb'])
        assert 'is_at_ub' in row and is_bin(row['is_at_ub']), 'row {}. is_at_ub key {}, value {}'.format(
            row_i, 'is_at_ub' in row, row['is_at_ub'])
        assert 'dual_sol_val' in row and is_an(row['dual_sol_val']), 'row {}. dual_sol_val key {}, value {}'.format(
            row_i, 'dual_sol_val' in row, row['dual_sol_val'])
        for ctype in ['linear', 'logicor', 'knapsack', 'setppc', 'varbound']:
            assert ctype in row and is_bin(row[ctype]), 'row {}. {} key: {}, value {}'.format(
                row_i, ctype, ctype in row, row[ctype])

    # Now we finally do this for the edges of our bipartite graph
    assert 'edges' in feature_dict and type(feature_dict['edges']) == list
    assert 'coefficients' in feature_dict and type(feature_dict['coefficients']) == list
    for i, edge in enumerate(feature_dict['edges']):
        coefficients_i = feature_dict['coefficients'][i]
        assert is_an(coefficients_i, True), 'edge {} has weight {}'.format(i, coefficients_i)
        # Our edge index is (column, row)
        assert len(edge) == 2 and edge[0] in feature_dict['cols'] and edge[1] in feature_dict['rows']

    return


def extract_features_from_dict(feature_dict):
    """
    Function for transforming the feature dictionary returned by our Branch Rule Feature Extractor.
    This could be more efficiently done without the dictionary, but being able to easily retrieve the more
    human readable dictionary is important.
    Args:
        feature_dict: A dictionary containing all feature information to encode a GNN data point

    Returns:
        The transformed dictionary into numpy arrays
    """

    ensure_features_are_correct(feature_dict)

    # There is a 'feature' in SCIP with how it prints MPS files. They contain all fixed variables. This means that
    # the GCNN would contain columns (variable nodes) that do not feature in any constraint as it has fixed bounds.
    # If a variable doesn't feature in any constraints, then setting it's value becomes trivial depending on its sign
    # in the objective, and if there's no sign then the variable is meaningless. We thus scan through all of the
    # constraints, and remove any variables that do not feature in any of them.

    # Get the columns that do feature in the constraints
    cols_featuring_in_cons = sorted(list(set([edge[0] for edge in feature_dict['edges']])))
    # Get the change in index that is going to happen for each col_i
    change_in_ids = {}
    # Initialise a counter that stores what the last col_i +1 was.
    last_col = 0
    # Initialise a counter that stores what the total missed columns have been
    num_fixed_cols = 0
    for col_i in cols_featuring_in_cons:
        # Remove all columns that appear in the difference between the last column featured and this one
        for col_j in range(last_col, col_i):
            del feature_dict['cols'][col_j]
        # Increment the number of fixed columns
        num_fixed_cols += col_i - last_col
        # Use the number of fixed columns at this point to represent the difference in indices that will be changed
        change_in_ids[col_i] = num_fixed_cols
        # Increment the last seen column index
        last_col = col_i + 1

    # Now remove all column information that appears after the final col_featuring_in_cons
    max_col_i = max([col_i for col_i in list(feature_dict['cols'].keys())])
    assert last_col > 0
    for col_i in range(last_col, max_col_i + 1):
        del feature_dict['cols'][col_i]

    # Now change the col_i's that feature in the edge indices
    for edge in feature_dict['edges']:
        edge[0] = edge[0] - change_in_ids[edge[0]]
        assert edge[0] >= 0

    # Now create the edge indices for the graph topology, the column feature matrix, the row feature matrix, and the
    # edge feature tensor!

    # First the edge indices. These are already nicely formatted. Our edges have the format (col_i, row_j)
    edge_indices = np.array([[edge[0], edge[1]] for edge in feature_dict['edges']])
    # Torch geometric needs the shape to be (2,n) for the edges, not (n,2). We thus need to swap axes
    edge_indices = np.swapaxes(edge_indices, 0, 1)

    # Now the non-zero coefficients in the rows.
    coefficients = np.array(feature_dict['coefficients'])
    # Need to fill in the dummy axis for the GNN input style. Turns (n,) shape into (n,1)
    coefficients = coefficients.reshape((len(coefficients), 1))

    # Now we retrieve the variable features (columns in our LP relaxation).
    # First we organise the static column features of the instance. These do not depend on the LP solution.
    binaries = np.array([feature_dict['cols'][col_i]['BINARY'] for col_i in feature_dict['cols']])
    integers = np.array([feature_dict['cols'][col_i]['INTEGER'] for col_i in feature_dict['cols']])
    continuous = np.array([feature_dict['cols'][col_i]['CONTINUOUS'] for col_i in feature_dict['cols']])
    implicit_ints = np.array([feature_dict['cols'][col_i]['IMPLINT'] for col_i in feature_dict['cols']])
    obj_coeffs = np.array([feature_dict['cols'][col_i]['obj_coeff'] for col_i in feature_dict['cols']])
    has_lbs = np.array([feature_dict['cols'][col_i]['has_lb'] for col_i in feature_dict['cols']])
    has_ubs = np.array([feature_dict['cols'][col_i]['has_ub'] for col_i in feature_dict['cols']])

    '''
    Non static column features. These features depend on LP solver
    sol_vals = np.array([feature_dict['cols'][col_i]['sol_val'] for col_i in feature_dict['cols']])
    basis_lower = np.array([feature_dict['cols'][col_i]['lower'] for col_i in feature_dict['cols']])
    basis_basic = np.array([feature_dict['cols'][col_i]['basic'] for col_i in feature_dict['cols']])
    basis_upper = np.array([feature_dict['cols'][col_i]['upper'] for col_i in feature_dict['cols']])
    basis_zero = np.array([feature_dict['cols'][col_i]['zero'] for col_i in feature_dict['cols']])
    red_costs = np.array([feature_dict['cols'][col_i]['red_cost'] for col_i in feature_dict['cols']])
    sol_fracs = np.array([feature_dict['cols'][col_i]['sol_frac'] for col_i in feature_dict['cols']])
    at_lbs = np.array([feature_dict['cols'][col_i]['is_at_lb'] for col_i in feature_dict['cols']])
    at_ubs = np.array([feature_dict['cols'][col_i]['is_at_ub'] for col_i in feature_dict['cols']])
    col_features = np.stack((sol_vals, binaries, integers, continuous, implicit_ints, obj_coeffs, has_lbs, has_ubs,
                            basis_lower, basis_basic, basis_upper, basis_zero, red_costs, sol_fracs, at_lbs, at_ubs),
                            axis=-1)
    '''

    # The above arrays contain all the features for our variables (columns). We now simply stack them and create
    # the feature vector for col_i by taking the i_th entry in all vectors
    col_features = np.stack((binaries, integers, continuous, implicit_ints, obj_coeffs, has_lbs, has_ubs), axis=-1)

    # Now we retrieve the constraint features (rows in our LP relaxation). Remember that these can be drastically
    # different to our original problem constraints as SCIP pre-solved the instance.
    obj_cosines = np.array([feature_dict['rows'][row_i]['obj_cosine'] for row_i in feature_dict['rows']])
    biases = np.array([feature_dict['rows'][row_i]['bias'] for row_i in feature_dict['rows']])
    linear_cons = np.array([feature_dict['rows'][row_i]['linear'] for row_i in feature_dict['rows']])
    logicor_cons = np.array([feature_dict['rows'][row_i]['logicor'] for row_i in feature_dict['rows']])
    knapsack_cons = np.array([feature_dict['rows'][row_i]['knapsack'] for row_i in feature_dict['rows']])
    setppc_cons = np.array([feature_dict['rows'][row_i]['setppc'] for row_i in feature_dict['rows']])
    varbound_cons = np.array([feature_dict['rows'][row_i]['varbound'] for row_i in feature_dict['rows']])

    '''
    row_at_lbs = np.array([feature_dict['rows'][row_i]['is_at_lb'] for row_i in feature_dict['rows']])
    row_at_ubs = np.array([feature_dict['rows'][row_i]['is_at_ub'] for row_i in feature_dict['rows']])
    dual_sol_vals = np.array([feature_dict['rows'][row_i]['dual_sol_val'] for row_i in feature_dict['rows']])
    row_features = np.stack((obj_cosines, biases, row_at_lbs, row_at_ubs, dual_sol_vals), axis=-1)
    '''

    # Similar to what we did to the columns, we just need to stack the above arrays into one larger array
    row_features = np.stack((obj_cosines, biases, linear_cons, logicor_cons, knapsack_cons, setppc_cons, varbound_cons),
                            axis=-1)

    return edge_indices, coefficients, col_features, row_features


def remove_previous_npy_files(transformed_problem_dir):

    for file in os.listdir(transformed_problem_dir):
        if file.endswith('.npy'):
            os.remove(os.path.join(transformed_problem_dir, file))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transformed_problem_dir', type=is_dir)
    parser.add_argument('temp_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('num_rand_seeds', type=int)
    args = parser.parse_args()

    # Remove all solve information from previous runs
    remove_previous_npy_files(args.transformed_problem_dir)
    remove_temp_files(args.temp_dir)
    # Make a subdirectory within the outfile_dir for this runs information
    args.outfile_dir = os.path.join(args.outfile_dir, 'feature_generation')
    if not os.path.isdir(args.outfile_dir):
        os.mkdir(args.outfile_dir)
    else:
        remove_slurm_files(args.outfile_dir)

    # Initialise a list of instances
    instance_names = set()

    for mps_file in os.listdir(args.transformed_problem_dir):
        # Extract the instance
        if mps_file.endswith('.mps'):
            instance_names.add(mps_file.split('__')[0])

    instance_names = list(instance_names)
    print('List of instances are: {}'.format(instance_names), flush=True)
    print('There are {} may instances'.format(len(instance_names)), flush=True)

    # Initialise the random seeds
    random_seeds = [random_seed for random_seed in range(1, args.num_rand_seeds + 1)]

    # Make sure all the appropriate instance files exist
    instance_paths = []
    for instance_name in instance_names:
        for random_seed in random_seeds:
            mps_file = get_filename(args.transformed_problem_dir, instance_name, random_seed, trans=True, root=False,
                                    sample_i=None, ext='mps')
            npy_file = get_filename(args.transformed_problem_dir, instance_name, random_seed, trans=True, root=False,
                                    sample_i=None, ext='npy')
            assert os.path.isfile(mps_file), 'File {} does not exist'.format(mps_file)
            assert not os.path.isfile(npy_file), 'File {} already exists'.format(npy_file)
            instance_paths.append(mps_file)

    for instance_name in instance_names:
        for random_seed in random_seeds:
            print('Generating features for {} with seed {}'.format(instance_name, random_seed), flush=True)
            generate_feature_vectors(instance_name, random_seed, args.transformed_problem_dir)
