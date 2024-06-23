import os
import subprocess
import itertools

# Define parameter combinations
generator_names = ["0002", "0011", "1005", "1016", "2006",
                   "2017", "5035", "5007", "5010", "1004"]

other_params = {
    'scanpath_generator_type': ["'random'", "'max_saliency'", "'percentile_saliency'", "'probabilistic_saliency'", "'inhibition_saliency'", "'inhibition_saliency'", "'inhibition_saliency'", "'inhibition_saliency'"],
    'equator_bias': [False, False, False, False, False, True, False, True],
    'fixation_distance': [False, False, False, False, False, False, True, True],
}

# Path to the configuration file and the main script file
config_file_path = 'config.py'
main_script_path = 'main.py'

param_length = len(other_params['scanpath_generator_type'])
assert param_length == len(other_params['equator_bias']) == len(other_params['fixation_distance']), "The parameter lists in other_params must have the same length."

# Function to modify the config.py file
def modify_config(generator_name, scanpath_generator_type, equator_bias, fixation_distance):
    with open(config_file_path, 'r') as file:
        config_lines = file.readlines()

    with open(config_file_path, 'w') as file:
        for line in config_lines:
            if line.startswith('generator_name_video'):
                file.write(f'generator_name_video = "{generator_name}"\n')
            elif line.startswith('scanpath_generator_type'):
                file.write(f'scanpath_generator_type = {scanpath_generator_type}\n')
            elif line.startswith('equator_bias'):
                file.write(f'equator_bias = {equator_bias}\n')
            elif line.startswith('fixation_distance'):
                file.write(f'fixation_distance = {fixation_distance}\n')
            else:
                file.write(line)

# Function to run the main script
def run_experiment(generator_name, params):
    modify_config(generator_name, params)
    result = subprocess.run(['python', main_script_path], capture_output=True, text=True)
    print(f"Output for {generator_name} with params {params}:\n{result.stdout}\n")

# Run the experiments
for generator_name in generator_names:
    for i in range(param_length):
        run_experiment(generator_name,
                       other_params['scanpath_generator_type'][i],
                       other_params['equator_bias'][i],
                       other_params['fixation_distance'][i])