import argparse
import numpy as np
import os
import torch
from GNN.GNN import GNNPolicy
from utilities import is_dir, read_feature_vector_files
from Slurm.train_neural_network import get_instances
from parameters import NUM_TORCH_SEEDS


def get_random_seeds(data_dir):
    files = os.listdir(data_dir)
    files = [file for file in files if file.endswith('.yml')]
    rand_seeds = set()
    for file in files:
        file_base_name = os.path.splitext(file)[0]
        rand_seed = file_base_name.split('__')[-1]
        rand_seeds.add(int(rand_seed))

    return sorted(list(rand_seeds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=is_dir)
    args = parser.parse_args()

    # Get the list of instances
    instances = get_instances(args.data_dir)

    # Get the random seeds
    random_seeds = get_random_seeds(args.data_dir)

    # The reference vector
    reference_vector = np.array([0.25, 0.25, 0.25, 0.25])

    # Torch seeds distance:
    torch_seed_distance = [0 for i in range(NUM_TORCH_SEEDS)]

    for torch_seed in range(len(torch_seed_distance)):
        # Create a new network for each new torch seed
        torch.manual_seed(torch_seed)
        neural_network = GNNPolicy()
        for instance in instances:
            for random_seed in random_seeds:
                edge_indices, coefficients, col_features, row_features = read_feature_vector_files(
                    args.data_dir, instance, random_seed, torch_output=True)
                cut_selector_params = neural_network.forward(edge_indices, coefficients, col_features, row_features)
                cut_selector_params = cut_selector_params.detach().numpy()
                distance = float(np.mean(np.abs(cut_selector_params - reference_vector)))
                torch_seed_distance[torch_seed] += distance
        torch_seed_distance[torch_seed] /= (len(instances) * len(random_seeds))
        print('Done with random seed {}'.format(torch_seed), flush=True)

    best_seed = int(np.argmin(np.array(torch_seed_distance)))

    print('Seed {} has smallest distance {}'.format(best_seed, torch_seed_distance[best_seed]))
    print('Average distance was: {}'.format(np.mean(np.array(torch_seed_distance))))
    print(torch_seed_distance)
