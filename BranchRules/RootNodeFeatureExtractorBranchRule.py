from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING, Branchrule
import numpy as np
import logging
import pdb


class RootNodeFeatureExtractor(Branchrule):

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.count = 0

    def relative_fractional(self, value):
        fractional_part = self.model.frac(value)
        return fractional_part if fractional_part <= 0.5 else 1 - fractional_part

    def extract_features(self):
        """
        Function for extracting features from the pyscipopt model. This needs to be called after solving the
        probing LP as we need the current LP information.
        Features used are a subset of those found in:
        Exact Combinatorial Optimization with Graph Convolutional Neural Networks (Gasse et.al)
        We use a subset of features as we are only interested in the root-node LP relaxation, so do not want any
        primal solution information or the age.
        Returns:
            Adds the bipartite graph representation and features to the branching rule.
        """

        self.features['rows'] = {}
        self.features['cols'] = {}

        # We first cycle through the variables. 
        cols = self.model.getLPColsData()
        ncols = self.model.getNLPCols()

        # Initialise the reduced cost list and objective coefficient list so we can later normalise our result
        red_costs = []
        obj_coefficients = []

        # Initialise the lbs and ubs list so we can later normalise our result
        lbs = []
        ubs = []

        for i in range(ncols):
            # Gets position of column in current LP, or -1 if it is not in LP
            col_i = cols[i].getLPPos()
            # Gets variable this column represents
            var = cols[i].getVar()

            self.features['cols'][col_i] = {}

            # Gets the LP solution value of the variable
            solval = var.getLPSol()

            # Gets lower bound and upper bound of column
            lb = cols[i].getLb()
            ub = cols[i].getUb()

            assert var.getLbGlobal() == lb, print(var.getLbGlobal(), lb)
            assert var.getUbGlobal() == ub, print(var.getUbGlobal(), ub)

            # sol_val represents where on the bound interval it lies
            # We use this instead of the direct LP sol value because the non-normalised val is too difficult to scale.
            # TODO: This measure is somewhat non-sensical for variables without a non-infinite ub / lb
            # self.features['cols'][col_i]['sol_val'] = solval
            self.features['cols'][col_i]['sol_val'] = (solval - lb) / (ub - lb + 1e-8)

            # Get the variable type
            for vtype in ['BINARY', 'INTEGER', 'CONTINUOUS', 'IMPLINT']:
                if var.vtype() == vtype:
                    self.features['cols'][col_i][vtype] = 1
                else:
                    self.features['cols'][col_i][vtype] = 0

            # Get the objective value coefficient of the column. Note that this is non-normalised!
            self.features['cols'][col_i]['obj_coeff'] = cols[i].getObjCoeff()
            obj_coefficients.append(self.features['cols'][col_i]['obj_coeff'])

            # Get the indicator on whether the column has an upper and lower bound
            # TODO: Compare if this indicator is better than the continuous approach
            # self.features['cols'][col_i]['has_lb'] = 0 if self.model.isInfinity(abs(lb)) else 1
            # self.features['cols'][col_i]['has_ub'] = 0 if self.model.isInfinity(abs(ub)) else 1
            # So 0-1 normalised bounds, and in the case of no bound, give it 2 for ub and -2 for lb
            self.features['cols'][col_i]['has_lb'] = lb if not self.model.isInfinity(abs(lb)) else None
            self.features['cols'][col_i]['has_ub'] = ub if not self.model.isInfinity(abs(ub)) else None
            lbs.append(lb if not self.model.isInfinity(abs(lb)) else 0)
            ubs.append(ub if not self.model.isInfinity(abs(ub)) else 0)

            # Get the basis status of the column in the LP
            for btype in ['lower', 'basic', 'upper', 'zero']:
                if cols[i].getBasisStatus() == btype:
                    self.features['cols'][col_i][btype] = 1
                else:
                    self.features['cols'][col_i][btype] = 0

            # Get the reduced cost of the variable in the LP. Note that this is currently non-normalised
            self.features['cols'][col_i]['red_cost'] = self.model.getVarRedcost(var)
            red_costs.append(self.features['cols'][col_i]['red_cost'])

            # Get how fractional the LP solution for this variable is
            self.features['cols'][col_i]['sol_frac'] = 2 * self.relative_fractional(solval)

            # Get whether the value at the LP solution is tight w.r.t the variable's bounds
            self.features['cols'][col_i]['is_at_lb'] = 1 if self.model.isEQ(solval, lb) else 0
            self.features['cols'][col_i]['is_at_ub'] = 1 if self.model.isEQ(solval, ub) else 0

        # Now normalise the reduced costs of the variable and the objective coefficients
        red_cost_norm = float(np.linalg.norm(red_costs))
        if red_cost_norm == 0:
            print('All variables have reduced cost 0')
            red_cost_norm = 1
        assert red_cost_norm > 0, 'All variables are in the basis'
        if red_cost_norm > 0:
            for col_i in self.features['cols']:
                self.features['cols'][col_i]['red_cost'] = self.features['cols'][col_i]['red_cost'] / red_cost_norm

        obj_coeff_norm = float(np.linalg.norm(obj_coefficients))
        assert obj_coeff_norm > 0, 'Problem is a feasibility problem, as all coefficients are 0 in the objective!'
        if obj_coeff_norm > 0:
            for col_i in self.features['cols']:
                self.features['cols'][col_i]['obj_coeff'] = self.features['cols'][col_i]['obj_coeff'] / obj_coeff_norm

        # We normalise the LB and UB values. -2 represents an infinite LB and +2 an infinite UB
        lb_norm = float(np.linalg.norm(lbs))
        ub_norm = float(np.linalg.norm(ubs))
        for col_i in self.features['cols']:
            if self.features['cols'][col_i]['has_lb'] is None:
                self.features['cols'][col_i]['has_lb'] = -2
            else:
                self.features['cols'][col_i]['has_lb'] = self.features['cols'][col_i]['has_lb'] / (1 + lb_norm)
                assert -1 <= self.features['cols'][col_i]['has_lb'] <= 1
        for col_i in self.features['cols']:
            if self.features['cols'][col_i]['has_ub'] is None:
                self.features['cols'][col_i]['has_ub'] = 2
            else:
                self.features['cols'][col_i]['has_ub'] = self.features['cols'][col_i]['has_ub'] / (1 + ub_norm)
                assert -1 <= self.features['cols'][col_i]['has_ub'] <= 1

        # We now cycle through the constraints
        rows = self.model.getLPRowsData()
        nrows = self.model.getNLPRows()

        # Initialise a dual_sol_val list so we can normalise our results later
        dual_sol_vals = []
        row_biases = []
        is_ranged_rows = []
        is_lhs_rows = []
        row_norms = []

        for i in range(nrows):
            # Gets position of row in current LP, or -1 if it is not in LP
            row_i = rows[i].getLPPos()
            self.features['rows'][row_i] = {}

            # Get the norm of the row
            norm = rows[i].getNorm()
            assert norm > 0

            # Note that: lhs <= activity + cst <= rhs
            lhs = rows[i].getLhs()
            rhs = rows[i].getRhs()
            cst = rows[i].getConstant()
            activity = self.model.getRowLPActivity(rows[i])

            assert not self.model.isInfinity(cst)

            # Get the objective parallelism
            self.features['rows'][row_i]['obj_cosine'] = self.model.getRowObjParallelism(rows[i])

            # Get the bias of the row
            lhss = (lhs - cst) if not self.model.isInfinity(lhs) else 0
            rhss = (rhs - cst) if not self.model.isInfinity(rhs) else 0
            if not self.model.isInfinity(lhs) and not self.model.isInfinity(rhs):
                is_ranged_rows.append(True)
                is_lhs_rows.append(False)
                row_biases.append(rhss)
            else:
                is_ranged_rows.append(False)
                if not self.model.isInfinity(lhs):
                    is_lhs_rows.append(True)
                    row_biases.append(-1 * lhss)
                else:
                    is_lhs_rows.append(False)
                    row_biases.append(rhss)

            # Sign here is irrelevant as it is only used for norm calculations of the bias
            row_non_zeros = rows[i].getVals()
            if is_ranged_rows[-1]:
                # In Gurobi, ranged rows are treated as equality constraints, with a new slack variable being added.
                # This slack variable has bounds [0, rhs - lhs].
                row_non_zeros.append(1)
            row_non_zeros.append(row_biases[-1])
            row_norms.append(float(np.linalg.norm(row_non_zeros)))
            assert row_norms[-1] > 0, 'Row has all zero coefficients and bias!'
            self.features['rows'][row_i]['bias'] = row_biases[-1] / row_norms[-1]

            # Get whether the row is tight at it's LB or UB
            self.features['rows'][row_i]['is_at_lb'] = 1 if self.model.isEQ(activity, lhs) else 0
            self.features['rows'][row_i]['is_at_ub'] = 1 if self.model.isEQ(activity, rhs) else 0

            # Get the dual solution value of the row in the LP. Note that this non-normalised
            self.features['rows'][row_i]['dual_sol_val'] = self.model.getRowDualSol(rows[i])
            dual_sol_vals.append(self.features['rows'][row_i]['dual_sol_val'])

        # Normalise the dual solution values
        dual_sol_norm = float(np.linalg.norm(dual_sol_vals))
        if dual_sol_norm > 0:
            for row_i in self.features['rows']:
                self.features['rows'][row_i]['dual_sol_val'] = self.features['rows'][row_i]['dual_sol_val'] / \
                                                               dual_sol_norm
        assert len(is_ranged_rows) == nrows
        assert len(is_lhs_rows) == nrows
        assert len(row_norms) == nrows

        # Now construct bipartite graph representation of the LP relaxation
        # We say LP relaxation here as SCIP internally can store our problem differently than the LP's state
        # Currently store the bipartite graph as a massive list of edges, with the indices (row_i, col_j)

        # Initialise the edge list and coefficient list
        self.features['edges'] = []
        self.features['coefficients'] = []
        for i in range(nrows):
            # Get the index of the row and all coefficients and columns with non-zero entries
            row_i = rows[i].getLPPos()
            row_cols = rows[i].getCols()
            row_vals = rows[i].getVals()
            # If we have a LHS constraint, we multiple the result by -1 to make a RHS constraint
            if is_lhs_rows[i]:
                row_vals = [-1 * row_val for row_val in row_vals]
            assert len(row_cols) == len(row_vals)
            # Cycle over the column and add the edge to our graph along with the corresponding coefficient
            for j, col in enumerate(row_cols):
                self.features['edges'].append([col.getLPPos(), row_i])
                self.features['coefficients'].append(row_vals[j] / row_norms[i])

        return

    def branchexeclp(self, allowaddcons):
        self.count += 1
        if self.count >= 2:
            logging.error('Dummy branch rule is called after root node and its first child')
            quit()
        assert allowaddcons

        assert not self.model.inRepropagation()
        assert not self.model.inProbing()
        self.model.startProbing()
        assert not self.model.isObjChangedProbing()
        self.model.constructLP()
        self.model.solveProbingLP()

        self.extract_features()

        self.model.endProbing()
        self.model.interruptSolve()

        # Make a dummy child. This branch rule should only be used at the root node!
        self.model.createChild(1, 1)
        return {"result": SCIP_RESULT.BRANCHED}
