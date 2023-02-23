import argparse
import os
import shutil
import random
from utilities import is_dir, remove_temp_files, get_instances, get_filename, get_random_seeds

TRAIN_TEST_SPLIT_RANDOM_SEED = 2022
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2

assert TRAIN_SPLIT + TEST_SPLIT == 1, 'Invalid distribution of TRAIN / TEST'


def split_instances_into_training_and_test(instance_dir, train_instance_dir, test_instance_dir):
    """
    Splits the downloaded instances into a training and test set.
    Args:
        instance_dir (dir): Directory containing the presolved instances
        train_instance_dir (dir): Directory in which some of the presolved instance files will be stored
        test_instance_dir (dir): Directory in which some of the presolved instance files will be stored

    Returns:
        Nothing. Just moves all the existing files appropriately
    """

    # Set the random seed
    random.seed(TRAIN_TEST_SPLIT_RANDOM_SEED)

    # Get the instance names and random seeds
    instances = get_instances(instance_dir)
    random_seeds = get_random_seeds(instance_dir)

    # We do the split of train-test instances as 80/20. Feel free to modify this in TRAIN_TEST_SPLIT
    train_instances = random.sample(instances, int(len(instances) * TRAIN_SPLIT))
    test_instances = [instance for instance in instances if instance not in train_instances]

    for instance_list, new_instance_dir in [(train_instances, train_instance_dir), (test_instances, test_instance_dir)]:
        for instance in instance_list:
            for rand_seed in random_seeds:
                instance_file = get_filename(instance_dir, instance, rand_seed, trans=True, root=False, sample_i=None,
                                             ext='mps')
                new_instance_file = get_filename(new_instance_dir, instance, rand_seed, trans=True, root=False,
                                                 sample_i=None, ext='mps')
                assert os.path.isfile(instance_file), 'Instance {} with seed {} does not exist'.format(instance,
                                                                                                       rand_seed)
                assert not os.path.isfile(new_instance_file), 'Instance {} seed {} already exists'.format(
                    instance, rand_seed)
                shutil.copy(instance_file, new_instance_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transformed_problem_dir', type=is_dir)
    parser.add_argument('train_instance_dir', type=is_dir)
    parser.add_argument('test_instance_dir', type=is_dir)
    args = parser.parse_args()

    # Make sure the instance directory only has instance files inside of it
    instance_file_strings = os.listdir(args.transformed_problem_dir)
    for instance_file_string in instance_file_strings:
        assert instance_file_string.endswith('.mps'), 'Instance {} in instance_dir, but not MPS file!'.format(
            instance_file_string)

    # Make sure that the test instance directory is empty
    remove_temp_files(args.test_instance_dir)
    remove_temp_files(args.train_instance_dir)

    # Call the main function
    split_instances_into_training_and_test(args.transformed_problem_dir, args.train_instance_dir,
                                           args.test_instance_dir)
