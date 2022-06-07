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
        Function for extracting features from the pyscipopt model.
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
            assert var.vtype() in ['BINARY', 'INTEGER', 'CONTINUOUS', 'IMPLINT']
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
            assert cols[i].getBasisStatus() in ['lower', 'basic', 'upper', 'zero']
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
        for col_i in self.features['cols']:
            self.features['cols'][col_i]['red_cost'] = self.features['cols'][col_i]['red_cost'] / red_cost_norm

        # Normalise the objective coefficients
        obj_coeff_norm = float(np.linalg.norm(obj_coefficients))
        assert obj_coeff_norm > 0, 'Problem is a feasibility problem, as all coefficients are 0 in the objective!'
        for col_i in self.features['cols']:
            self.features['cols'][col_i]['obj_coeff'] = self.features['cols'][col_i]['obj_coeff'] / obj_coeff_norm

        # We normalise the LB and UB values. -2 represents an infinite LB and +2 an infinite UB
        # We normalise the bounds using the infinity norm as opposed to the two norm here.
        bound_norm_type = np.inf
        # bound_norm_type = None
        lb_norm = float(np.linalg.norm(lbs, ord=bound_norm_type))
        ub_norm = float(np.linalg.norm(ubs, ord=bound_norm_type))
        if lb_norm == 0:
            lb_norm = 1
        if ub_norm == 0:
            ub_norm = 1
        for col_i in self.features['cols']:
            if self.features['cols'][col_i]['has_lb'] is None:
                self.features['cols'][col_i]['has_lb'] = -2
            else:
                self.features['cols'][col_i]['has_lb'] = self.features['cols'][col_i]['has_lb'] / lb_norm
                assert -1 <= self.features['cols'][col_i]['has_lb'] <= 1
        for col_i in self.features['cols']:
            if self.features['cols'][col_i]['has_ub'] is None:
                self.features['cols'][col_i]['has_ub'] = 2
            else:
                self.features['cols'][col_i]['has_ub'] = self.features['cols'][col_i]['has_ub'] / ub_norm
                assert -1 <= self.features['cols'][col_i]['has_ub'] <= 1

        # We now cycle through the constraints
        rows = self.model.getLPRowsData()
        nrows = self.model.getNLPRows()

        # Initialise lists that contain information on each row. Ranged rows will be manually split into two rows
        dual_sol_vals = []
        row_rhss = []
        row_lhss = []
        is_ranged_row = []
        is_lhs_row = []
        row_norms = []

        # As our bipartite graph representation is identical when the columns and rows are permuted, we do not need
        # to keep track of the LPPos. Doing so is possible, but makes adding ranged rows difficult and changes
        # the structure anyway. So we simply ignore this.
        for i in range(nrows):

            # Get the actual row_i index that has been changed to accommodate ranged rows
            row_i = i + sum(is_ranged_row)
            # Initialise the dictionary for the row
            self.features['rows'][row_i] = {}

            # Get the norm of the row
            norm = rows[i].getNorm()
            assert norm > 0

            # Note that: lhs <= activity + cst <= rhs
            lhs = rows[i].getLhs()
            rhs = rows[i].getRhs()
            cst = rows[i].getConstant()
            activity = self.model.getRowLPActivity(rows[i])

            assert not self.model.isInfinity(abs(cst))

            # Get the rhs / lhs of the row. We manually transform all rows into single rhs rows
            lhss = (lhs - cst) if not self.model.isInfinity(abs(lhs)) else 0
            rhss = (rhs - cst) if not self.model.isInfinity(abs(rhs)) else 0
            # Handle ranged rows as two rhs constraints. The first is the normal rhs, and the next is the flipped lhs
            if not self.model.isInfinity(abs(lhs)) and not self.model.isInfinity(abs(rhs)):
                self.features['rows'][row_i + 1] = {}
                is_ranged_row.append(True)
                is_lhs_row.append(False)
                row_rhss.append(rhss)
                row_lhss.append(lhss)
            else:
                is_ranged_row.append(False)
                if not self.model.isInfinity(abs(lhs)):
                    is_lhs_row.append(True)
                    row_rhss.append(-1 * lhss)
                    row_lhss.append(None)
                else:
                    is_lhs_row.append(False)
                    row_rhss.append(rhss)
                    row_lhss.append(None)

            # Sign here is irrelevant as it is only used for norm calculations of the bias
            row_non_zeros = rows[i].getVals()
            if is_ranged_row[-1]:
                row_norms.append(float(np.linalg.norm(row_non_zeros + [row_rhss[-1]])))
                row_norms.append(float(np.linalg.norm(row_non_zeros + [row_lhss[-1]])))
            else:
                assert row_lhss[-1] is None
                if row_rhss[-1] is not None:
                    row_non_zeros.append(row_rhss[-1])
                row_norms.append(float(np.linalg.norm(row_non_zeros)))

            # Make sure that the norms are strictly greater than zero
            assert row_norms[-1] > 0, 'Row has all zero coefficients and bias!'
            if is_ranged_row[-1]:
                assert row_norms[-2] > 0, 'Row has all zero coefficients and bias!'

            if is_ranged_row[-1]:
                assert row_rhss[-1] is not None and row_lhss[-1] is not None
                self.features['rows'][row_i]['bias'] = row_rhss[-1] / row_norms[-2]
                self.features['rows'][row_i + 1]['bias'] = row_lhss[-1] / row_norms[-1]
            else:
                assert row_rhss[-1] is not None
                self.features['rows'][row_i]['bias'] = row_rhss[-1] / row_norms[-1]

            # Get the objective parallelism
            self.features['rows'][row_i]['obj_cosine'] = self.model.getRowObjParallelism(rows[i])
            if is_ranged_row[-1]:
                # As the parallelism measure takes an absolute value, the values are the same for the rhs and lhs row
                self.features['rows'][row_i + 1]['obj_cosine'] = self.model.getRowObjParallelism(rows[i])

            # Get whether the row is tight at its LB or UB
            if is_ranged_row[-1]:
                # The only way a converted ranged row is tight at both the ub and lb is if lhs and rhs are equal
                self.features['rows'][row_i]['is_at_lb'] = 1 if lhs == rhs else 0
                self.features['rows'][row_i]['is_at_ub'] = 1 if self.model.isEQ(activity, rhs) else 0
                self.features['rows'][row_i + 1]['is_at_lb'] = 1 if self.model.isEQ(activity, lhs) else 0
                self.features['rows'][row_i + 1]['is_at_ub'] = 1 if lhs == rhs else 0
            else:
                self.features['rows'][row_i]['is_at_lb'] = 1 if self.model.isEQ(activity, lhs) else 0
                self.features['rows'][row_i]['is_at_ub'] = 1 if self.model.isEQ(activity, rhs) else 0

            # Get the dual solution value of the row in the LP. Note that this non-normalised
            self.features['rows'][row_i]['dual_sol_val'] = self.model.getRowDualSol(rows[i])
            if is_ranged_row[-1]:
                # For the flipped lhs of ranged rows we simply take the same dual value
                self.features['rows'][row_i + 1]['dual_sol_val'] = self.model.getRowDualSol(rows[i])
            dual_sol_vals.append(self.features['rows'][row_i]['dual_sol_val'])
            if is_ranged_row[-1]:
                dual_sol_vals.append(self.features['rows'][row_i + 1]['dual_sol_val'])

            # Get the constraint handler responsible for the row
            cons_type_row = rows[i].getConsOriginConshdlrtype()
            # Note that our experiments only handled linear, logicor, knapsack, setppc, and varbound constraints
            # This can be extended quite easily, but shouldn't if none of those constraint type exist over the instances
            assert cons_type_row in ['linear', 'logicor', 'knapsack', 'setppc', 'varbound']
            for cons_type in ['linear', 'logicor', 'knapsack', 'setppc', 'varbound']:
                if rows[i].getConsOriginConshdlrtype() == cons_type:
                    self.features['rows'][row_i][cons_type] = 1
                    if is_ranged_row[-1]:
                        self.features['rows'][row_i + 1][cons_type] = 1
                else:
                    self.features['rows'][row_i][cons_type] = 0
                    if is_ranged_row[-1]:
                        self.features['rows'][row_i + 1][cons_type] = 0

        # Normalise the dual solution values
        dual_sol_norm = float(np.linalg.norm(dual_sol_vals))
        if dual_sol_norm > 0:
            for row_i in self.features['rows']:
                self.features['rows'][row_i]['dual_sol_val'] = self.features['rows'][row_i]['dual_sol_val'] / \
                                                               dual_sol_norm
        assert len(is_ranged_row) == nrows
        assert len(is_lhs_row) == nrows

        # Now construct bipartite graph representation of the LP relaxation
        # We say LP relaxation here as SCIP internally can store our problem differently than the LP's state
        # Currently store the bipartite graph as a massive list of edges, with the indices (row_i, col_j)

        # Initialise the edge list and coefficient list
        self.features['edges'] = []
        self.features['coefficients'] = []
        # Keep a separate counter for the row_i index used in our dictionary
        row_i = 0
        for i in range(nrows):
            # Get the index of the row and all coefficients and columns with non-zero entries
            row_cols = rows[i].getCols()
            row_vals = rows[i].getVals()
            # If we have a LHS constraint, we multiple the result by -1 to make a RHS constraint
            if is_lhs_row[i]:
                assert not is_ranged_row[i]
                row_vals = [-1 * row_val for row_val in row_vals]
            assert len(row_cols) == len(row_vals)
            # Cycle over the column and add the edge to our graph along with the corresponding coefficient
            for j, col in enumerate(row_cols):
                self.features['edges'].append([col.getLPPos(), row_i])
                self.features['coefficients'].append(row_vals[j] / row_norms[row_i])
            # Increment the row_i counter
            row_i += 1
            # If it is a ranged row, add the flipped lhs constraint
            if is_ranged_row[i]:
                row_vals = [-1 * row_val for row_val in row_vals]
                # Cycle over the column and add the edge to our graph along with the corresponding coefficient
                for j, col in enumerate(row_cols):
                    self.features['edges'].append([col.getLPPos(), row_i])
                    self.features['coefficients'].append(row_vals[j] / row_norms[row_i])
                row_i += 1

        return

    def branchexeclp(self, allowaddcons):
        self.count += 1
        if self.count >= 2:
            logging.error('Dummy branch rule is called after root node and its first child')
            quit()
        assert allowaddcons

        # Assert that the model is not doing anything funny
        assert not self.model.inRepropagation()
        assert not self.model.inProbing()

        # Extract the features of the model()
        self.extract_features()

        # Interrupt the solve. We only wanted features from this, we never wanted to actually branch.
        self.model.interruptSolve()

        # Make a dummy child. This branch rule should only be used at the root node!
        self.model.createChild(1, 1)
        return {"result": SCIP_RESULT.BRANCHED}
