#! /usr/bin/env python
import argparse
from utilities import is_dir, is_file, build_scip_model, get_filename, str_to_bool
import parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transformed_problem_dir', type=is_dir)
    parser.add_argument('instance_path', type=is_file)
    parser.add_argument('sol_path', type=is_file)
    parser.add_argument('instance', type=str)
    parser.add_argument('rand_seed', type=int)
    parser.add_argument('trans_sol', type=str_to_bool)
    args = parser.parse_args()

    # Generate the instance post pre-solve and print out the transformed model
    sol_path = args.sol_path if args.trans_sol else None
    scip = build_scip_model(args.instance_path, 1, args.rand_seed, True, True, False, False, False, True,
                            time_limit=parameters.PRESOLVE_TIME_LIMIT, sol_path=sol_path)

    if args.trans_sol:
        # The original solution is not always feasible in the transformed space. We thus disable dual pre-solve
        # We only do this for the MIPLIB solutions, as in the other case we generate the solution ourselves.
        scip.setParam('misc/allowstrongdualreds', False)
        scip.setParam('misc/allowweakdualreds', False)
        scip.setParam('presolving/dualagg/maxrounds', 0)
        scip.setParam('presolving/dualcomp/maxrounds', 0)
        scip.setParam('presolving/dualinfer/maxrounds', 0)
        scip.setParam('presolving/dualsparsify/maxrounds', 0)

    # Put the instance through pre-solving only. We run optimize though as we additionally want our solution transformed
    scip.optimize()

    # If the instance hit the time-limit, then we can simply ignore it. Note that this includes a single root LP solve
    if scip.getStatus() == 'timelimit':
        scip.freeProb()
        quit()
    if scip.getStatus() == 'optimal':
        scip.freeProb()
        quit()

    # Get the file_name of the transformed instance that we're going to write it out to
    transformed_file_name = get_filename(args.transformed_problem_dir, args.instance, args.rand_seed, trans=True,
                                         root=False, sample_i=None, ext='mps')

    # Get the file_name of the transformed solution that we're going to write it out to
    transformed_sol_name = get_filename(args.transformed_problem_dir, args.instance, args.rand_seed, trans=True,
                                        root=False, sample_i=None, ext='sol')

    # Write the actual transformed instance file
    scip.writeProblem(filename=transformed_file_name, trans=True)

    # In the case of using MIPLIB solutions, we want to print out the pre-loaded transformed solution
    if args.trans_sol:
        # We manually construct the solutions, as it is possible that during presolve and the root-solve that a better
        # solution that's infeasible in transformed space has been found, and writeBestTransSol would produce an error
        sols = scip.getSols()
        trans_sol = scip.createSol()
        for var in scip.getVars(transformed=True):
            var_val = scip.getSolVal(sols[0], var)
            scip.setSolVal(trans_sol, var, var_val)
        scip.writeTransSol(trans_sol, filename=transformed_sol_name)

    scip.freeProb()
