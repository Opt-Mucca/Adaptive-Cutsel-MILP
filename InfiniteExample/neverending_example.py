from pyscipopt import Model, SCIP_PARAMSETTING
import numpy as np
import time


def score(cut, objective, lambda_):
    """
    The function used to score individual cuts. It follows the rule:
    lambda * integer_support + (1 - lambda) * objective_parallelism
    Args:
        cut (list): The list of coefficients for the cut, including zeros. We don't need the RHS values for this score
        objective (list): The list of coefficients in the objective, including zeros.
        lambda_ (float / int): The lambda we will use in our equation, determines how we preference the ratio
    Returns:
        The score of the given cut
    """
    assert type(cut) == list and len(cut) == 3
    num_non_zeros = sum([1 for coefficient in cut if coefficient != 0])
    num_non_zero_integers = sum([1 for coefficient in [cut[0], cut[2]] if coefficient != 0])
    integer_support = num_non_zero_integers / num_non_zeros
    assert 0 <= integer_support <= 1

    objective_parallelism = abs(np.dot(cut, objective) / (np.linalg.norm(cut) * np.linalg.norm(objective)))
    assert 0 <= objective_parallelism <= 1

    return (lambda_ * integer_support) + ((1 - lambda_) * objective_parallelism)


def get_lambda_range(a, d):
    """
    Gets the range of lambda values for a given (a,d) pair, that would result in the GC being added, and the
    MIP then being solved.
    Args:
        a (float / int): The value of a we used in our MIP construction
        d (float / int): The value of d we used in our MIp construction
    Returns:
        The lower and upper bound values of the interval.
    """
    lambda_lower = (2 * (606 * (-32401 + 220 * np.sqrt(20301)) - 606 * a ** 2 + 120 * d * (
            -31411 + 211 * np.sqrt(20301) + 10 * (-151 + np.sqrt(20301)) * d) + 12 * a * (
                                 101 * (-110 + np.sqrt(20301)) + 10 * (-101 + np.sqrt(20301)) * d) + np.sqrt(
        20301) * np.sqrt((101 + a ** 2 + d * (20 + d)) * (3272501 - 22220 * np.sqrt(20301) + 101 * a ** 2 + 20 * d * (
            31411 - 211 * np.sqrt(20301) - 10 * (-151 + np.sqrt(20301)) * d) + a * (
                                                                  -202 * (-110 + np.sqrt(20301)) - 20 * (
                                                                  -101 + np.sqrt(20301)) * d))))) / (
                           505 * (-76409 + 528 * np.sqrt(20301)) + 5555 * a ** 2 + 24 * a * (
                           101 * (-110 + np.sqrt(20301)) + 10 * (-101 + np.sqrt(20301)) * d) + d * (
                                   -7403300 + 50640 * np.sqrt(20301) + (-355633 + 2400 * np.sqrt(20301)) * d))

    lambda_upper = (-73203 + 660 * np.sqrt(402) + (-609 + 6 * np.sqrt(402)) * a ** 2 + 60 * (
            -220 + np.sqrt(402) - 10 * d) * d + 6 * a * (
                            -421 + 111 * np.sqrt(402) + 10 * (-2 + np.sqrt(402)) * d) + np.sqrt(402) * np.sqrt(-(
            (101 + a ** 2 + d * (20 + d)) * (
            -24401 + 220 * np.sqrt(402) + (-203 + 2 * np.sqrt(402)) * a ** 2 + 20 * (
            -220 + np.sqrt(402) - 10 * d) * d + a * (
                    -842 + 222 * np.sqrt(402) + 20 * (-2 + np.sqrt(402)) * d))))) / (
                           -59669 + 660 * np.sqrt(402) + (-475 + 6 * np.sqrt(402)) * a ** 2 + 2 * (
                           -5260 + 30 * np.sqrt(402) - 233 * d) * d + 6 * a * (
                                   -421 + 111 * np.sqrt(402) + 10 * (-2 + np.sqrt(402)) * d))

    return lambda_lower, lambda_upper


def get_eps_value(num_iterations):
    """
    This is an example function that returns the values of a strictly increasing sequence with limit 0.1
    The sequence is (1 - (1/n)) / 10
    Args:
        num_iterations: The index number of the sequence you're after
    Returns:
        The value of the sequence at index num_iterations
    """

    return (1 - (1 / (num_iterations + 1))) / 10


def get_max_a_value(d):
    """
    This function gets the maximum value of a for a given d that still allows the existence of a lambda value
    that would score GC the as the best scoring cut
    Args:
        d (float / int): The value of d that we're analysing
    Returns:
        The maximum value a can take while still being finitely solvable via pure separation and our separators
        with only one cut being added per round
    """
    max_a = ((-6767 * np.sqrt(2)) - (27068 * np.sqrt(101)) + (22220 * np.sqrt(201)) - (2680 * np.sqrt(101) * d) +
             (2020 * np.sqrt(201) * d)) / ((6767 * np.sqrt(2)) - (202 * np.sqrt(201)))
    return max_a


def create_model(a, d, cuts):
    """
    The function to create the SCIP model. The base MIP has the following form:
    min x_1 - (10+d) x_2 - a x_3
    s.t -0.5 x_2 + 3 x_3 <= 0
        -x_3 <= 0
        -0.5 x_1 + 0.5 x_2 -3.5 x_3 <= 0
        0.5 x_1 + 1.5 x_3 <= 0.5
        x_1, x_3 Binary, x_2 Real

    Due to issues with redundancy checks in SCIP, and the interactions of limits and model resolving,
    we simply create the SCIP model each time we resolve it. This is inefficient, but remains exceedingly quick
    due to the small size of our model.

    In addition to the base model, all cuts we pass are added so we can recreate the model at each stage of the solving
    process. The cuts are given in the form [b_1, b_2, b_3, b_0], where they make the constraint:
    b_1 x_1 + b_2 x_2 + b_3 x_3 <= b_0
    Args:
        a (float / int): The value of a we use in our model construction
        d (float / int): The value of d we use in our model construction
        cuts (list): The list of cuts that we add directly to our model

    Returns:
        The SCIP model object and a list of the variables
    """

    # Create the base SCIP model object
    scip = Model()

    # Add the variables to the SCIP model
    x_1 = scip.addVar(name='x_1', vtype='I', lb=-1, ub=1)
    x_2 = scip.addVar(name='x_2', vtype='C')
    x_3 = scip.addVar(name='x_3', vtype='B')

    # Now add the set of base constraints
    scip.addCons(-0.5 * x_2 + 3 * x_3 <= 0)
    scip.addCons(-1 * x_3 <= 0)
    scip.addCons(-0.5 * x_1 + 0.5 * x_2 - 3.5 * x_3 <= 0)
    scip.addCons(0.5 * x_1 + 1.5 * x_3 <= 0.5)

    # Now add the objective to the model
    scip.setObjective(x_1 - ((10 + d) * x_2) - (a * x_3), sense='minimize')

    # Now add the appropriate parameters to disable everything but our added separator
    # Disable pre-solving
    scip.setPresolve(SCIP_PARAMSETTING.OFF)
    # Disable heuristics
    scip.setHeuristics(SCIP_PARAMSETTING.OFF)
    # Disable separators
    scip.setSeparating(SCIP_PARAMSETTING.OFF)
    # Set the node limit to 1
    scip.setParam('limits/nodes', 1)
    # Disable all propagation at the root node
    scip.setParam('propagating/maxroundsroot', 0)
    # Set a branch rule that is does use strong branching, that way to now child nodes are pre-evaluated
    scip.setParam('branching/leastinf/priority', 10000000)

    # Add the cuts that we have found to be highest scoring in previous iterations
    for cut in cuts:
        scip.addCons(cut[0] * x_1 + cut[1] * x_2 + cut[2] * x_3 <= cut[3])

    # We hide the output for tidiness. Remove this if you wish.
    scip.hideOutput()

    return scip, [x_1, x_2, x_3]


if __name__ == '__main__':

    # Set the values of a, d, and lambda you'd like to use for this run
    a_val = 4
    d_val = 0.5
    lambda_val = 0.540

    # Print some interesting statements that will predict the outcome of the run
    max_a_val = get_max_a_value(d_val)
    lambda_val_lower, lambda_val_upper = get_lambda_range(a_val, d_val)
    print('You have chosen a: {}, d: {}, lambda: {}'.format(a_val, d_val, lambda_val))
    if a_val > max_a_val:
        print('For the given d, a above {} is not finitely solvable for any lambda. Terminating experiment'.format(
            max_a_val
        ))
        quit()
    print('For a: {}, d: {}, lambda between [{}, {}] is needed to finitely solve'.format(a_val, d_val, lambda_val_lower,
                                                                                         lambda_val_upper))
    lambda_val_in_range = lambda_val_lower <= lambda_val <= lambda_val_upper
    print('Does given lambda: {} lie in this range: {}'.format(lambda_val, lambda_val_in_range))
    if lambda_val > lambda_val_upper + 0.0000001:
        print('The selected lambda val is above the GC lambda UB. Expect ISC cuts to be added each round')
    elif lambda_val < lambda_val_lower - 0.0000001:
        print('The selected lambda val is below the GC lambda LB. Expect OPC cuts to be added each round')
    elif lambda_val_lower + 0.0000001 < lambda_val < lambda_val_upper - 0.0000001:
        print('The selected lambda val is in the GC lambda range. Expect a single GC cut, and an optimal solution')
    else:
        print('Lambda {} too close to boundary of range [{}, {}]. Cannot guarantee numeric stability'.format(
            lambda_val,
            lambda_val_lower,
            lambda_val_upper)
        )

    # Initialise the data structure containing the history of highest scoring cuts
    applied_cuts = []
    # Create the base SCIP model
    model, variables = create_model(a_val, d_val, applied_cuts)

    # Create the objective coefficient list for later score calculations
    objective_coefficients = [1, -(10 + d_val), -1 * a_val]

    # Do the initial optimise call
    model.optimize()
    # Print the solution:
    print('Solution: {}'.format([var.getLPSol() for var in variables]))
    # Initialise the number of iterations counter that we use to find epsilon_n
    iteration_i = 1
    # Also create a maximum number of iterations. We do this as eventually our problem becomes non-computer friendly
    max_iterations = 1000

    # Now start the main while loop. Our break condition is if our solution is found to be optimal, which will always
    # happen when the root LP solve is integer feasible. This should only occur when GC is added!
    while model.getStatus() != 'optimal' and iteration_i < max_iterations:

        # Generate the coefficient vectors of our three cuts: OPC, ISC, and GC
        opc_coefficients = [-1, 10, 0]
        isc_coefficients = [-1, 0, 1]
        gc_coefficients = [-10, 10, 1]

        # Get the lower and upper bounds on the finite termination range.
        lower_bound, upper_bound = get_lambda_range(a_val, d_val)

        # Now get the RHS values of the cuts
        # First the RHS value of the opc cut. Remember that if ISC has been applied then the RHS needs to be changed
        if lambda_val > upper_bound and iteration_i > 1:
            opc_rhs = 30.5 - (31 * get_eps_value(iteration_i))
        else:
            opc_rhs = 30.5 - get_eps_value(iteration_i)
        # Second the isc RHS and the gc RHS. Both do not depend on the lambda value
        isc_rhs = 1 - get_eps_value(iteration_i)
        gc_rhs = 0

        # The three candidate cuts are now fully created.
        # Generate the scores for each cut. The score does not depend on the RHS value!!!
        opc_score = score(opc_coefficients, objective_coefficients, lambda_val)
        isc_score = score(isc_coefficients, objective_coefficients, lambda_val)
        gc_score = score(gc_coefficients, objective_coefficients, lambda_val)

        # We can now check if our calculations were correct. Specifically, the highest scoring cut is related to lambda
        if lambda_val > upper_bound + 0.0000001:
            assert isc_score > opc_score and isc_score > gc_score, 'ISC {}, OPC {}, GC {}'.format(isc_score, opc_score,
                                                                                                  gc_score)
            print('Lambda: {} larger than lambda UB {}. ISC the best scoring cut, will be added'.format(lambda_val,
                                                                                                        upper_bound))
            applied_cuts.append(isc_coefficients + [isc_rhs])
        elif lambda_val < lower_bound - 0.0000001:
            assert opc_score > isc_score and opc_score > gc_score, 'ISC {}, OPC {}, GC {}'.format(isc_score, opc_score,
                                                                                                  gc_score)
            print('Lambda: {} smaller than lambda LB {}. OPC the best scoring cut, will be added'.format(lambda_val,
                                                                                                         lower_bound))
            applied_cuts.append(opc_coefficients + [opc_rhs])
        elif lower_bound + 0.0000001 < lambda_val < upper_bound - 0.0000001:
            assert gc_score > isc_score and gc_score > opc_score, 'ISC {}, OPC {}, GC {}'.format(isc_score, opc_score,
                                                                                                 gc_score)
            print('Lambda: {} between [{}, {}]. GC the best scoring cut, will be added'.format(lambda_val, lower_bound,
                                                                                               upper_bound))
            applied_cuts.append(gc_coefficients + [gc_rhs])
        else:
            print('Lambda {} too close to boundary of range [{}, {}]. Cannot guarantee numeric stability'.format(
                lambda_val,
                lower_bound,
                upper_bound)
            )

        # Print statements on cut-added
        print('Cut {}x_1 + {}x_2 + {}x_3 <= {}'.format(applied_cuts[-1][0], applied_cuts[-1][1],
                                                       applied_cuts[-1][2], applied_cuts[-1][3]))
        lhs = sum([variables[i].getLPSol() * applied_cuts[-1][i] for i in range(0, 3)])
        print('Previous solution in this constraint is: {} <= {}'.format(lhs, applied_cuts[-1][3]))

        # Free the SCIP instance and remove all solution process attached data
        model.freeProb()

        # Now re-solve the model with the new cut
        model, variables = create_model(a_val, d_val, applied_cuts)
        model.optimize()

        # Print the solution
        print('Solution: {}'.format([var.getLPSol() for var in variables]))

        # We place a small sleep here as the loop is incredibly fast, and reading things may be difficult
        time.sleep(0.4)

        # Increment the number of iterations
        iteration_i += 1
