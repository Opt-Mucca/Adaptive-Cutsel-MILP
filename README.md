# Adaptive Cut Selection in Mixed-Integer Linear Programming

If this software was used for academic purposes, please cite our paper with the below information:

`@misc{turner2022adaptive,
      title={Adaptive Cut Selection in Mixed-Integer Linear Programming}, 
      author={Mark Turner and Thorsten Koch and Felipe Serrano and Michael Winkler},
      year={2022},
      eprint={2202.10962},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}`

- Install Guide

Requirements: Python 3.6 / Ubuntu (18.04 / 20.04) / Mathematica 20 (Probably installs on other Ubuntu versions, and if
you change the virtual environment a bit it should work on Python 3.{7,8,9}. Mathematica 18 should also be sufficient). 
We use SLURM <https://slurm.schedmd.com/overview.html> as a job manager. All calls go through a central function 
however, and in theory SLURM could be replaced by python's default multiprocessing package. 

Run the bash script init_venv. If you don't use bash, configure the shebang (first line of the script) to be your 
shell interpreter. 

`./init_venv`

After installing the virtual environment, make sure to always activate it with the script printed beneath. This is so
your python path is appended and files at different directory levels can import from each other.

`source ./set_venv`

Now go and install SCIP from <https://www.scipopt.org/index.php#download>. For Ubuntu / Debian, the .sh installer is 
the easiest choice if you don't want to configure it yourself). The cut selector plugin, which features heavily in 
this research is available from SCIP 8.0+.

You can test if SCIP is installed by locating `/bin/scip` and calling it through the command line. SCIP should 
hopefully open.

One then needs to install PySCIPOpt <https://github.com/scipopt/PySCIPOpt>. I would recommend following the 
`INSTALL.md` guide. Make sure to have set your environment variable pointing to SCIP! You can test if this has been
properly installed by running one of the tests, or by trying to import `Model`. This research was done on the branch
`mt/cut_selector_plugin`, but is available from PySCIPOpt version 4.1.0+.

- How to run the software

We use Nohup <https://en.wikipedia.org/wiki/Nohup> to run all of our jobs and to capture output of the main function
calls. It also allows jobs to be started through a SHH connection. Using this is not necessary however, so feel free 
to use whatever software you prefer. An example call to redirect output to `nohup/nohup.out` and to run the process in
the background would be

`nohup python dir/example.py > nohup/nohup.out &`

##### 1
We assume the instances that are going to be used are located in `Instances/Instances`. Feel free to throw your
own instances in that directory. Alternatively, one can run `Instances/get_instances.py`, which will download 
all MIPLIB instances from the website <https://miplib.zib.de/>.

An example run: `python Instances/get_instances.py`

##### 2

We will generate the standard data associated with each of the instances using
`Slurm/generate_standard_data.py`. Standard data in this context is 
a `.sol` file for every instance, a `.yml` file containing solving quality measures (e.g. solve_time) when run with 
default cut selector parameter values, and an additional `.yml` file for a run with default settings restricted to the 
root node. Both `.yml` files have an accompanying `.log` file showing all output from SCIP. 
Additionally there is a `.stats` file for each run with statistics from SCIP. While generating this data,
we also filter out undesirable instances. Change the methods we use to filter out those instances as you wish! 
This function also puts all problems through presolve and outputs the instance as a transformed problem. We do this
as we want to remove redundancy from the formulations, and mimic the actual solving process, but don't want
to run presolve each time we read in a problem. All output from this function (including now transformed `.mps` files
are placed into `transformed_problems/`.)

An example run: `nohup python Slurm/generate_standard_data.py Instances/Instances/ Instances
/Solutions/ transformed_problems/ experiments/temp_files/ slurm_outfiles/ 3 > nohup/generate_standard_data.out &`

##### 3

We will generate the feature vectors that create our bipartite graph representation of a MILP. 
This representation and the extraction code we use was inspired by <https://github.com/ds4dm/learn2branch>.
The feature extraction is done by disabling everything but branching, and then including the branching rule
`BranchRules/RootNodeFeatureExtractorBranchRule.py`. Browse the method if you're interested in the exact features and
how they're retrieved. Note that by default we do not save the non-static features, but that can be rectified by
uncommenting the appropriate lines.
Calling this function will create `coefficients.npy, col_features.npy, edge_indices.npy, row_features.npy` for each
instance, where these files are used to construct the input into our graph neural network. 
An example call to run this would be:

An example run: `nohup python Slurm/generate_feature_vectors.py transformed_problems/ experiments/temp_files/ 
slurm_outfiles/ 3 > nohup/generate_features.out &`

##### 4

To both filter out undesirable instances and to see the potential gain by using adaptive parameter choices,
one can run `Slurm/parameter_sweep.py`. This will delete some instances and their solve information from
`transformed_problems/`, and will create
the file `transformed_problems/potential_improvements.yaml`. Note that this must be run after 
`Slurm/generate_standard_data.py` else there will be no .sol files and standard data to compare against. 

An example run: `nohup python Slurm/parameter_sweep.py transformed_problems/ experiments/temp_files/ 
slurm_outfiles/ True > nohup/grid_search.out &`

##### 5 (optional)

To find the random seed we will use to initialise our neural network, we can run the script
`scripts/random_seed_finder.py`. This will load all of our instances and random SCIP seeds,
and find the random torch seed that minimises the output distance to the vector [0.25,0.25,0.25,0.25].
This initialisation process does not have to be used, but we believe that it gives the neural network more freedom
on deciding which parameters are important through learning.

An example run: `nohup python scripts/random_seed_finder.py transformed_problems/ > nohup/random_seed_finder.out &`

##### 6

Now we can train a neural network! We should have all the appropriate files in `transformed_problems/`. 
A call to this function produces a saved neural network file `transformed_problems/actor.pt` that corresponds to an 
object of `GNN/GNN.py::GNNPolicy`. The function centrally creates the samples we use in training, and updates 
the neural network, but each call to a SCIP solve is an individual job that is run on the cluster through SLURM and 
saves its results to a file. The various measures of SCIPs solution quality and of the neural networks training 
itself are saved to tensorboard. The final boolean input decides if you want to iteratively recreate a base 
`GNN/GNN.py::GNNPolicy` and overfit on individual instances, creating `transformed_problems/instance_name.pt` for
each instance. This occurs when True is used, as opposed to False, which trains a single neural network
`transformed_problems/actor.pt` that trains using batches over the entire instance set.

Here 500 is the number of epochs (times each instance is seen), 0.025 is the relative batch size
(So each batch is 2.5% of the instance set size), 20 is the number of samples from the normal distribution taken
using the REINFORCE algorithm, 953 is the random seed found from  the above (5 (optional)), and False tells us
that we want to train a single neural network. 
An example run: `nohup python Slurm/train_neural_network.py transformed_problems/ experiments
/runs/ experiments/temp_files/ None slurm_outfiles/ 500 0.025 20 953 False > nohup/train_network.out &`

##### 7

To evaluate a trained network and do a single run with no updates, one can run 
`Slurm/evaluate_trained_network.py`. This uses the mean of the multivariate normal distributions as the singular sample
for each call to SCIP. The results of the run are also output into tensorboard. 

An example run: `nohup python Slurm/generate_standard_data.py transformed_problems/ experiments/runs/ 
experiments/temp_files/ transformed_problems/actor.pt slurm_outfiles/ True > nohup/nohup.out &`


