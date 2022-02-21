import wget
import csv
import time
import os
import requests

if __name__ == '__main__':

    assert os.path.isdir('Instances/') and os.path.isfile('Instances/collection_set.csv')
    assert os.path.isdir('Instances/Instances') and os.path.isdir('Instances/Solutions')

    # Read the csv containing all instance names and tags
    instance_descriptions = []
    with open('Instances/collection_set.csv', 'r') as s:
        reader = csv.reader(s, delimiter=',')
        for row in reader:
            instance_descriptions.append(row)

    # Get instances that do not contain tags: feasibility, numerics, infeasible, no_solution
    valid_rows = []
    num_instances = len(instance_descriptions)
    num_feasibility_instances = 0
    num_numerics_instances = 0
    num_infeasible_instances = 0
    num_no_solution_instances = 0
    num_unbounded_instances = 0
    for row_i, row in enumerate(instance_descriptions):
        if row_i == 0:
            continue
        if 'feasibility' in row[-1]:
            num_feasibility_instances += 1
            continue
        if 'numerics' in row[-1]:
            num_numerics_instances += 1
            continue
        if 'infeasible' in row[-1]:
            num_infeasible_instances += 1
            continue
        if 'no_solution' in row[-1]:
            num_no_solution_instances += 1
            continue
        if 'Unbounded' in row[-2]:
            num_unbounded_instances += 1
            continue
        valid_rows.append(row_i)

    instances = []
    for row_i in valid_rows:
        instances.append(instance_descriptions[row_i][0])

    # Download the instances
    num_no_miplib_solution_instances = 0
    for instance in instances:
        # Download the instance and solution from the MIPLIB website directly
        mps_url = 'https://miplib.zib.de/WebData/instances/{}.mps.gz'.format(instance)
        sol_url = 'https://miplib.zib.de/downloads/solutions/{}/1/{}.sol.gz'.format(instance, instance)

        headers = requests.head(sol_url).headers
        if 'text/html' in headers['Content-Type']:
            num_no_miplib_solution_instances += 1
            continue

        mps_file = 'Instances/Instances/{}.mps.gz'.format(instance)
        wget.download(mps_url, mps_file)
        time.sleep(0.1)

        sol_file = 'Instances/Solutions/{}.sol.gz'.format(instance)
        wget.download(sol_url, sol_file)
        time.sleep(0.1)

    print('num instances: {}'.format(num_instances))
    print('num filtered instances with feasibility flag: {}'.format(num_feasibility_instances))
    print('num filtered instances with numerics flag: {}'.format(num_numerics_instances))
    print('num filtered instances with infeasible flag: {}'.format(num_infeasible_instances))
    print('num filtered instances with no_solution flag: {}'.format(num_no_solution_instances))
    print('num unbounded instances with no valid solution: {}'.format(num_unbounded_instances))
    print('num no available MIPLIB solution {}'.format(num_no_miplib_solution_instances))

