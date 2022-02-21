"""File containing the settings for the experiments you want to perform.
Each parameter here affects different bits of the experiments. The individual comments outline how.
"""

# If you want to use the MIPLIB solution instead of one found after a 10 minute solve, set this to True
# In the paper this was set to False
USE_MIPLIB_SOLUTIONS = False

# The time-limit that filters instances that take longer than this time to presolve. Time in seconds.
# Note that this time includes the root node LP solve if USE_MIPLIB_SOLUTIONS is True. In the paper this was set to 300
PRESOLVE_TIME_LIMIT = 300

# The time-limit that filters instances that take too long find any feasible solution. Time in seconds.
# Note this is ignored if USE_MIPLIB_SOLUTIONS is set to True. In the paper this was set to 600
SOL_FIND_TIME_LIMIT = 600

# The number of cut-selection rounds we will apply. Note that this is only used when restricted to the root node.
# In the paper this was set to 50
NUM_CUT_ROUNDS = 50

# The number of cuts that are attempted to be applied at each cut-selection round. Note that if not enough cuts are
# presented to the cut-selector, then all those presented are applied. This was set to 10 for the paper.
NUM_CUTS_PER_ROUND = 10

# The fraction of NUM_CUT_ROUNDS * NUM_CUTS_PER_ROUND cuts that need to be added over all random seeds to consider
# the instance good enough to keep. This was set to 0.5 in the paper
MIN_NUM_CUT_RATIO = 0.5

# The time-limit that filters instances that take too long to do x rounds of cut-selection and one round of presolve.
# This time is in seconds. In the paper this was set 20
ROOT_SOLVE_TIME_LIMIT = 20

# The time-limit that is used to compare runs under standard conditions against using the trained cut-selector.
# This time is in seconds. In the paper this was set to 600
FULL_SOLVE_TIME_LIMIT = 600

# The number of torch random seeds that we want to trial for learning. This is used to generate all torch seeds
# between [0, n-1], and the one that minimises the distance to cut-selector parameters [0.25, 0.25, 0.25, 0.25]
# is chosen. In the paper this was set to 1000
NUM_TORCH_SEEDS = 1000

# Make sure to set this before you begin any runs. This is the SLURM queue that'll be used
SLURM_QUEUE = 'PUT QUEUE ID HERE'




