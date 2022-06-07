#! /usr/bin/env python
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import Cutsel
import random

class FixedAmountCutsel(Cutsel):

    def __init__(self, num_cuts_per_round=20, min_orthogonality_root=0.9,
                 min_orthogonality=0.9, dir_cutoff_dist_weight=0.0, efficacy_weight=1.0, int_support_weight=0.1,
                 obj_parallel_weight=0.1):
        super().__init__()
        self.num_cuts_per_round = num_cuts_per_round
        self.min_orthogonality_root = min_orthogonality_root
        self.min_orthogonality = min_orthogonality
        self.dir_cutoff_dist_weight = dir_cutoff_dist_weight
        self.int_support_weight = int_support_weight
        self.obj_parallel_weight = obj_parallel_weight
        self.efficacy_weight = efficacy_weight
        random.seed(42)

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """
        This is the main function used to select cuts. It must be named cutselselect and is called by default when
        SCIP performs cut selection if the associated cut selector has been included (assuming no cutsel with higher
        priority was called successfully before). This function aims to add self.num_cuts_per_round many cuts to
        the LP per round, prioritising the highest ranked cuts. It adds the highest ranked cuts, filtering by
        parallelism. In the case when not enough cuts are added and all the remaining cuts are too parallel,
        we simply add those with the highest score.
        @param cuts: These are the optional cuts we get to select from
        @type cuts: List of pyscipopt rows
        @param forcedcuts: These are the cuts that must be added
        @type forcedcuts: List of pyscipopt rows
        @param root: Boolean for whether we're at the root node or not
        @type root: Bool
        @param maxnselectedcuts: Maximum number of selected cuts
        @type maxnselectedcuts: int
        @return: Dictionary containing the keys 'cuts', 'nselectedcuts', result'. Warning: Cuts can only be reordered!
        @rtype: dict
        """
        # Initialise number of selected cuts and number of cuts that are still valid candidates
        n_cuts = len(cuts)
        nselectedcuts = 0

        # Get the number of cuts that we will select this round.
        num_cuts_to_select = min(maxnselectedcuts, max(self.num_cuts_per_round - len(forcedcuts), 0), n_cuts)

        # Initialises parallel thresholds. Any cut with 'good' score can be at most good_max_parallel to a previous cut,
        # while normal cuts can be at most max_parallel. (max_parallel >= good_max_parallel)
        if root:
            max_parallel = 1 - self.min_orthogonality_root
            good_max_parallel = max(0.5, max_parallel)
        else:
            max_parallel = 1 - self.min_orthogonality
            good_max_parallel = max(0.5, max_parallel)

        # Generate the scores of each cut and thereby the maximum score
        max_forced_score, forced_scores = self.scoring(forcedcuts)
        max_non_forced_score, scores = self.scoring(cuts)

        good_score = max(max_forced_score, max_non_forced_score)

        # This filters out all cuts in cuts who are parallel to a forcedcut.
        for forced_cut in forcedcuts:
            n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, forced_cut, cuts,
                                                                scores, max_parallel, good_max_parallel, good_score)

        if maxnselectedcuts > 0 and num_cuts_to_select > 0:
            while n_cuts > 0:
                # Break the loop if we have selected the required amount of cuts
                if nselectedcuts == num_cuts_to_select:
                    break
                # Re-sorts cuts and scores by putting the best cut at the beginning
                cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                nselectedcuts += 1
                n_cuts -= 1
                n_cuts, cuts, scores = self.filter_with_parallelism(n_cuts, nselectedcuts, cuts[nselectedcuts -1], cuts,
                                                                    scores, max_parallel, good_max_parallel,
                                                                    good_score)

            # So far we have done the algorithm from the default method. We will now enforce choosing the highest
            # scored cuts from those that were previously removed for being too parallel.
            # Reset the n_cuts counter
            n_cuts = len(cuts) - nselectedcuts
            for remaining_cut_i in range(nselectedcuts, num_cuts_to_select):
                cuts, scores = self.select_best_cut(n_cuts, nselectedcuts, cuts, scores)
                nselectedcuts += 1
                n_cuts -= 1

        return {'cuts': cuts, 'nselectedcuts': nselectedcuts,
                'result': SCIP_RESULT.SUCCESS}

    def scoring(self, cuts):
        """
        Scores each cut in cuts. The current rule is a weighted sum combination of the efficacy,
        directed cutoff distance, integer support, and objective function parallelism.
        @param cuts: The list of cuts we want to find scores for
        @type cuts: List of pyscipopt rows
        @return: The max score over all cuts in cuts as well as the individual scores
        @rtype: Float and List of floats
        """
        # initialise the scoring of each cut as well as the max_score
        scores = [0] * len(cuts)
        max_score = 0.0

        # We require this check as getBestSol() may return the lp solution, which is not a valid primal solution
        sol = self.model.getBestSol() if self.model.getNSols() > 0 else None

        # Separate into two cases depending on whether the directed cutoff distance contributes to the score
        if sol is not None:
            for i in range(len(cuts)):
                int_support = self.int_support_weight * \
                              self.model.getRowNumIntCols(cuts[i]) / cuts[i].getNNonz()
                obj_parallel = self.obj_parallel_weight * self.model.getRowObjParallelism(cuts[i])
                efficacy = self.model.getCutEfficacy(cuts[i])
                if cuts[i].isLocal():
                    score = self.dir_cutoff_dist_weight * efficacy
                else:
                    score = self.model.getCutLPSolCutoffDistance(cuts[i], sol)
                    score = self.dir_cutoff_dist_weight * max(score, efficacy)
                efficacy *= self.efficacy_weight
                score += obj_parallel + int_support + efficacy
                score += 1e-4 if cuts[i].isInGlobalCutpool() else 0
                score += random.uniform(0, 1e-6)
                max_score = max(max_score, score)
                scores[i] = score
        else:
            for i in range(len(cuts)):
                int_support = self.int_support_weight * \
                              self.model.getRowNumIntCols(cuts[i]) / cuts[i].getNNonz()
                obj_parallel = self.obj_parallel_weight * self.model.getRowObjParallelism(cuts[i])
                efficacy = (self.efficacy_weight + self.dir_cutoff_dist_weight) * self.model.getCutEfficacy(cuts[i])
                score = int_support + obj_parallel + efficacy
                score += 1e-4 if cuts[i].isInGlobalCutpool() else 0
                score += random.uniform(0, 1e-6)
                max_score = max(max_score, score)
                scores[i] = score

        return max_score, scores

    def filter_with_parallelism(self, n_cuts, nselectedcuts, cut, cuts, scores, max_parallel, good_max_parallel,
                                good_score):
        """
        Filters the given cut list by any cut_iter in cuts that is too parallel to cut. It does this by moving the
        parallel cut to the back of cuts, and decreasing the indices of the list that are scanned over.
        For the main portion of our selection we then never touch these cuts. In the case of us wanting to
        forcefully select an amount which is impossible under this filtering method however, we simply select the
        remaining highest scored cuts from the supposed untouched cuts.
        @param n_cuts: The number of cuts that are still viable candidates
        @type n_cuts: int
        @param nselectedcuts: The number of cuts already selected
        @type nselectedcuts: int
        @param cut: The cut which we will add, and are now using to filter the remaining cuts
        @type cut: pyscipopt row
        @param cuts: The list of cuts
        @type cuts: List of pyscipopt rows
        @param scores: The scores of each cut
        @type scores: List of floats
        @param max_parallel: The maximum allowed parallelism for non good cuts
        @type max_parallel: Float
        @param good_max_parallel: The maximum allowed parallelism for good cuts
        @type good_max_parallel: Float
        @param good_score: The benchmark of whether a cut is 'good' and should have it's allowed parallelism increased
        @type good_score: Float
        @return: The now number of viable cuts, the complete list of cuts, and the complete list of scores
        @rtype: int, list of pyscipopt rows, list of pyscipopt rows
        """
        # Go backwards through the still viable cuts.
        for i in range(nselectedcuts + n_cuts - 1, nselectedcuts - 1, -1):
            cut_parallel = self.model.getRowParallelism(cut, cuts[i])
            # The maximum allowed parallelism depends on the whether the cut is 'good'
            allowed_parallel = good_max_parallel if scores[i] >= good_score else max_parallel
            if cut_parallel > allowed_parallel:
                # Throw the cut to the end of the viable cuts and decrease the number of viable cuts
                cuts[nselectedcuts + n_cuts - 1], cuts[i] = cuts[i], cuts[nselectedcuts + n_cuts - 1]
                scores[nselectedcuts + n_cuts - 1], scores[i] = scores[i], scores[nselectedcuts + n_cuts - 1]
                n_cuts -= 1

        return n_cuts, cuts, scores

    def select_best_cut(self, n_cuts, nselectedcuts, cuts, scores):
        """
        Moves the cut with highest score which is still considered viable (not too parallel to previous cuts) to the
        front of the list. Note that 'front' here still has the requirement that all added cuts are still behind it.
        @param n_cuts: The number of still viable cuts
        @type n_cuts: int
        @param nselectedcuts: The number of cuts already selected to be added
        @type nselectedcuts: int
        @param cuts: The list of cuts themselves
        @type cuts: List of pyscipopt rows
        @param scores: The scores of each cut
        @type scores: List of floats
        @return: The re-sorted list of cuts, and the re-sorted list of scores
        @rtype: List of pyscipopt rows, list of floats
        """
        # Initialise the best index and score
        best_pos = nselectedcuts
        best_score = scores[nselectedcuts]
        for i in range(nselectedcuts + 1, nselectedcuts + n_cuts):
            if scores[i] > best_score:
                best_pos = i
                best_score = scores[i]
        # Move the cut with highest score to the front of the still viable cuts
        cuts[nselectedcuts], cuts[best_pos] = cuts[best_pos], cuts[nselectedcuts]
        scores[nselectedcuts], scores[best_pos] = scores[best_pos], scores[nselectedcuts]
        return cuts, scores