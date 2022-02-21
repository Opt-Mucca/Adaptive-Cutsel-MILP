# Instance collector

The file `get_instances.py` is run simply by `python Instances/get_instances.py`. Alternatively one can run
`nohup python Instances/get_instances.py > nohup/nohup.out &`. It scrapes the MIPLIB 2017 website for
all instances from the collection set. Downloaded instances are placed in `Instances/Instances/`. Using
`collection_set.csv`, instances containing flags numerics, infeasible, feasibility, and
no solution are filtered out and not downloaded. The function also downloads the first solution of the instance
available through the MIPLIB website, and places it in `Instances/Solutions`.
No code after this point is specific to the MIPLIB dataset, so your
own instances can be placed in `Instances/Instances/` to later train on instead.