import argparse
import torch
from tqdm import tqdm
import numpy as np
import sbibm
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--task", 
                    "-d", 
                    help="string for SBI task", 
                    default="sir")
parser.add_argument(
    "--seed", 
    "-s", 
    help="int for random seed to be fixed", default=45, type=int
)
parser.add_argument(
    "--n_replicates",
    "-nr",
    help="int for number of replications to make",
    default=30,
    type=int,
)
parser.add_argument(
    "--B",
    "-B",
    help="int for number of observed X to be generated",
    default=10000,
    type=int,
)


if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task
seed = args.seed
n_replica = args.n_replicates
B = args.B

# Set the random seed for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# Load the SBI task, simulator, and prior
task = sbibm.get_task(task_name)
simulator = task.get_simulator()
prior = task.get_prior()


# Generate the observed data and saving it in a list
X_obs_list = []
theta_obs_list = []

# creating the directory to save the results
save_dir = f"Results/X_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# checking if the checkpoint file already exists
checkpoint_path = os.path.join(save_dir, 
                               f"{task.name}_X_checkpoint.pkl"
                               )

checkpoint_path_theta = os.path.join(save_dir, 
                               f"{task.name}_theta_checkpoint.pkl"
                               )
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as checkpoint_file:
        X_obs_list = pickle.load(checkpoint_file)

    with open(checkpoint_path_theta, "rb") as checkpoint_file:
        theta_obs_list = pickle.load(checkpoint_file)
    i = len(X_obs_list)  # Start from the first empty entry
else:
    i = 0


# simulating theta and X observed
for j in tqdm(range(i, n_replica)):
    # generating theta and X observed
    theta_obs = prior(num_samples=B)
    X_obs = simulator(theta_obs)

    # saving the generated data
    X_obs_list.append(X_obs)
    theta_obs_list.append(theta_obs)
    # Save a checkpoint after processing each X
    with open(checkpoint_path, "wb") as checkpoint_file:
        pickle.dump(X_obs_list, checkpoint_file)

    with open(checkpoint_path_theta, "wb") as checkpoint_file:
        pickle.dump(theta_obs_list, checkpoint_file)  

# Saving samples
with open(os.path.join(save_dir, f"{task.name}_X_samples.pkl"), "wb") as f:
    pickle.dump(X_obs_list, f)

with open(os.path.join(save_dir, f"{task.name}_theta_samples.pkl"), "wb") as f:
    pickle.dump(theta_obs_list, f)

# removing checkpoint file
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    os.remove(checkpoint_path_theta)
