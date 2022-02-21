from pyscipopt import Model, Conshdlr, SCIP_RESULT, SCIP_PRESOLTIMING, SCIP_PROPTIMING
"""
This is a dummy constraint handler that is used to force the number of separation rounds that we want.
It checks if the number of separation rounds has been hit, and if it hasn't then it sends the solver back to solve
the same node.
To use this you must set the enforce priority to be a positive value so it is called before branching.
You must also set need constraints to be False otherwise it will not be called.
"""


class RepeatSepaConshdlr(Conshdlr):

    def __init__(self, model, max_separation_rounds):
        super().__init__()
        self.model = model
        self.max_separation_rounds = max_separation_rounds

    # fundamental callbacks
    def consenfolp(self, constraints, nusefulconss, solinfeasible):

        if self.model.getNSepaRounds() <= self.max_separation_rounds and self.model.getNNodes() == 1:
            return {'result': SCIP_RESULT.SOLVELP}
        else:
            return {'result': SCIP_RESULT.FEASIBLE}

    def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
        return {"result": SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        return
