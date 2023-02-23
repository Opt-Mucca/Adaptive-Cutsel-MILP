import numpy as np
import os
import argparse
import shutil
import random
from utilities import is_dir, str_to_bool, remove_slurm_files

NN_VERIFICATION_SEED = 2022
NUM_INSTANCES = 1000


def copy_instance_files(full_instance_dir, partial_instance_dir):
    """
    Copy instance files from full_instance_dir to partial_instance_dir.
    Remove all files from partial_instance_dir first.
    Args:
        full_instance_dir (dir): Directory with the originally downloaded instances
        partial_instance_dir (dir): Directory where we will place a subset of files

    Returns:
        Absolutely nothing
    """

    instances = os.listdir(full_instance_dir)
    random.seed(NN_VERIFICATION_SEED)
    random_instances = random.sample(instances, NUM_INSTANCES)

    remove_slurm_files(partial_instance_dir)

    for instance in random_instances:
        shutil.copy(os.path.join(full_instance_dir, instance), os.path.join(partial_instance_dir, instance))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('full_instance_dir', type=is_dir)
    parser.add_argument('partial_instance_dir', type=is_dir)
    args = parser.parse_args()

    assert args.partial_instance_dir.endswith('Instances/') or args.partial_instance_dir.endswith('Instances')

    copy_instance_files(args.full_instance_dir, args.partial_instance_dir)
