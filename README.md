# Adaptive Cut Selection in Mixed-Integer Linear Programming

If this software was used for academic purposes, please cite our paper with the below information:

`
@article{turner2023adaptive,
     author = {Mark Turner and Thorsten Koch and Felipe Serrano and Michael Winkler},
     title = {Adaptive {Cut} {Selection} in {Mixed-Integer} {Linear} {Programming}},
     journal = {Open Journal of Mathematical Optimization},
     eid = {5},
     publisher = {Universit\'e de Montpellier},
     volume = {4},
     year = {2023},
     doi = {10.5802/ojmo.25},
     language = {en},
     url = {https://ojmo.centre-mersenne.org/articles/10.5802/ojmo.25/}
}`

## Install Guide

Requirements: Python 3.8 / Ubuntu (18.04 / 20.04) / Mathematica 20 (Probably installs on other Ubuntu versions, and if
you change the virtual environment a bit it should work on Python 3.{6,7,9,10}. Mathematica 18 should also be
 sufficient). 
We use SLURM <https://slurm.schedmd.com/overview.html> as a job manager. All calls go through a central function 
however, and in theory SLURM could be replaced by python's default multiprocessing package. 

Run the bash script init_venv. If you don't use bash, configure the shebang (first line of the script) to be your 
shell interpreter. 

`./init_venv`

After installing the virtual environment, make sure to always activate it with the script printed beneath. This is so
your python path is appended and files at different directory levels can import from each other.

`source ./set_venv`

Now go and install SCIP from <https://www.scipopt.org/index.php#download>. For Ubuntu / Debian, the .sh installer is 
the easiest choice if you don't want to configure it yourself. The cut selector plugin, which features heavily in 
this research is only available from SCIP 8.0+.

You can test if SCIP is installed by locating `/bin/scip` and calling it through the command line. SCIP should 
hopefully open.

One then needs to install PySCIPOpt <https://github.com/scipopt/PySCIPOpt>. I would recommend following the 
`INSTALL.md` guide. Make sure to have set your environment variable pointing to SCIP! You can test if this has been
properly installed by running one of the tests, or by trying to import `Model`.

## Run Guide

We use Nohup <https://en.wikipedia.org/wiki/Nohup> to run all of our jobs and to capture output of the main function
calls. It also allows jobs to be started through a SHH connection. Using this is not a requirement, so feel free 
to use whatever software you prefer. An example call to redirect output to `nohup/nohup.out` and to run the process in
the background would be

`nohup python dir/example.py > nohup/nohup.out &`

For all the following calls, we assume all data and result files are from a directory indexed by the data set name. E
.g. `Instances/MIPLIB2017/Instances` or `outfiles/NN-Verification/run_170`.

- Download Instances

To obtain the MIPLIB2017 instances we have created an automatic script. Simply run:

`python Instances/MIPLIB2017/get_instances.py`

This script will download all MIPLIB2017 instances from the website <https://miplib.zib.de/>. Note that 
the csv file `collection_set.csv` is used to filter out numerically troublesome instances etc. The best solution
from the MIPLIB website is also downloaded for each instance (although this is not used in our experiment by default). 

For the NN-Verification data set, please download all instances at the website 
<https://github.com/deepmind/deepmind-research/tree/master/neural_mip_solving>. Due to computational constraints,
the entire data set may be too large. For this reason, we have a script that randomly selects a subset of the instances.
One can run this script with the following:

`python Instances/NN-Verification/select_instance_subset.py path/to/downloaded/instances path/to/place/subset/instances`

- Generate standard data

We will generate the standard data associated with each of the instances using
`Slurm/generate_standard_data.py`. Standard data in this context is 
a `.sol` file for every instance, and a `.yml` file containing solving quality measures (e.g. solve_time) when run with 
default cut selector parameter values. The `.yml` files have accompanying `.log` files showing all output from SCIP. 
Additionally there are `.stats` files for each run with statistics from SCIP. While generating this data
we also filter out undesirable instances. Change the methods we use to filter out those instances as you wish! 
This function also puts all problems through presolve and outputs the instance (a `.mps` file) as a transformed problem
. We do this as we want to remove redundancy from the formulations and mimic the actual solving process, but don't want
to run presolve each time we read in a problem. The files created by this function are placed into several different
directories. 

An example run:

`nohup python Slurm/generate_standard_data.py instance_dir solution_dir transformed_instance_dir
 transformed_solution_dir root_results_dir tree_results_dir temp_file_dir outfile_dir num_rand_seeds True False 
 True > nohup/MIPLIB2017/generate_standard_data.out &`

Note that `solution_dir` here can be empty, and we will then find the best solutions after x-minutes. The last arguments
are just for the user to inform the script if files are in compressed format, lp or mps, and if the solution
 directory is empty.
 
 - Generate GCNN input
 
 We will now generate the GCNN feature vectors that create our bipartite graph representation of a MILP. 
This representation and the extraction code we use was inspired by <https://github.com/ds4dm/learn2branch>.
The feature extraction is done by disabling everything but branching, and then including the branching rule
`BranchRules/RootNodeFeatureExtractorBranchRule.py`. Browse the method if you're interested in the exact features and
how they're retrieved. Note that by default we do not save the non-static features, but that can be changed by
uncommenting the appropriate lines.
Calling this function will create `coefficients.npy, col_features.npy, edge_indices.npy, row_features.npy` for each
instance, where these files are used to construct the input into our graph neural network. 
An example call to run this would be:

An example run: `nohup python Slurm/generate_feature_vectors.py transformed_instance_dir/ feature_dir/ temp_file_dir
 outfile_dir num_rand_seeds >  nohup/MIPLIB2017/generate_features.out &`

- Perform a grid search

The following is a guide for how to perform a grid search of the cut selector parameter space. This script
generates all the following parameter combinations:

\sum_{i=1}^{4} \lambda_{i} = 1, | \lambda_{i} = \frac{\beta_{i}}{10}, \beta_{i} \in
 \mathbb{N}, \quad \forall i \in \{1,2,3,4\}

It stores the results of all parameters over each instance. It also stores the best parameter choice. In the case
that the best performing parameter is near identical to the worst, and in the case that many parameter choices are
 the best, we remove the instance from our data set for later experiments. The two files creates are:
 
`results/Instance-Set/grid_search.yaml` and `results/Instance-Set/all_grid_runs.yaml`
 
An example run is:

`nohup python Slurm/parameter_sweep.py transformed_instance_dir/ transformed_solution_dir/ feature_dir
 default_root_results_dir default_tree_results_dir final_results_dir temp_file_dir outfile_dir True > 
 nohup/MIPLIB2017/grid_search.out &`
 
The last argument of this script is whether or not we want to run root node experiments. We have to pass both
the root node default results and tree default results for the instance filtering methods. 

- Generate random seed (optional)

To find the random seed we will use to initialise our neural network, we can run the script
`scripts/random_seed_finder.py`. This will load all of our instances for all our SCIP seeds,
and find the random torch seed that minimises the output distance to the vector [0.25,0.25,0.25,0.25].
This initialisation process does not have to be used, but we believe that it gives the neural network more freedom
on deciding which parameters are important through learning. For further explanation on this choice, please see the
 paper.

An example run: `nohup python scripts/random_seed_finder.py transformed_instances/ features/ > nohup
/MIPLIB2017/random_seed_finder.out &`

- Run SMAC (standard ML approach)

Before we run our method, it is interesting to see how standard methods perform for this learning challenge. 
To this end we run SMAC, see the website: <https://www.automl.org/automated-algorithm-design/algorithm-configuration
/smac/>. The function that SMAC tries to manimise is the primal-dual difference relative to that produced by default
parameters over all instance and seed combinations. So after each prediction from SMAC it runs the predicted
parameter choice over all instance-seed pairs. An example run is:
  
`nohup python Slurm/smac_runs.py training_instance_dir test_instance_dir solution_dir default_results_dir 
results_this_experiment_dir temp_file_dir outfile_dir 250 667`

The last two arguments are the number of epochs and the random seed. For our experiments we used the random seed 2022.
 

- Train the GCNN

We should now have all the appropriate data for training a GCNN. The goal of the following call will be to produce
a saved GCNN file `results_to_somewhere/actor.pt` that corresponds to an object of `GNN/GNN.py:GNNPolicy`. Note that
the GCNN is periodically saved, but that for our results only the final GCNN is used. The various measures of SCIPs
solution quality and of the neural networks training itself are saved to tensorboard. An example call is:

`nohup python Slurm/train_neural_network.py train_instance_dir test_instance_dir solution_dir feature_dir
 default_results_dir results_this_run_dir tensorboard_dir temp_file_dir None outfile_dir 250 0.1 20 667 False > 
 nohup/MIPLIB2017/train_network.out &`
 
In this call, `None` is the previous network, which in case of wanting to pause training or to preload another
network can be a path to `saved_actor.pt`. `250` is the number of epochs, `0.1` is the relative batch size (10% of
the training set per batch), `20` is the number of samples from the normal distribution taken using the REINFORCE
algorithm, `667` is the random seeds found from our script above, and `False` is because
we don't want to train invidiaul networks per instance (an experiment done in the original paper version). 

- Evaluate trained GCNN

After training the network, one might want to evaluate the results on a different data set, or check the results
again. This is possible using the script below. Note that the mean of the multivariate normal distribution is used
as the singular sample for each call to SCIP. Tensorboard can be used to visualise the results of these runs too.

`nohup python Slurm/evaluate_trained_network.py instance_dir/ solution_dir feature_dir default_results_dir
 results_this_run_dir tensorboard_dir temp_file_dir path_to_gcnn outfile_dir True > nohup/MIPLIB2017/evaluate_network
 .out &`

The final argument `True` is if we want to evaluate our runs when restircting to the root node or not.

#### Thanks for Reading!! I hope this code helps you with your research. Please feel free to send any issues.


