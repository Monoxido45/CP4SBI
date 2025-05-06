import argparse
import torch
from tqdm import tqdm
import numpy as np
import sbibm
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--task", "-d", help="string for SBI task", default="two_moons")
parser.add_argument(
    "--seed", "-s", help="int for random seed to be fixed", default=45, type=int
)
parser.add_argument(
    "--n_samples",
    "-n",
    help="int for number of posterior samples to be generated for each X",
    default=1000,
    type=int,
)
parser.add_argument(
    "--n_x",
    "-nx",
    help="int for number of observed X to be generated",
    default=500,
    type=int,
)


if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task
seed = args.seed
n_samples = args.n_samples
n_x = args.n_x
# Set the random seed for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# Load the SBI task, simulator, and prior
task = sbibm.get_task(task_name)
simulator = task.get_simulator()
prior = task.get_prior()


# simulating theta and X observed
theta_obs = prior(num_samples=n_x)
X_obs = simulator(theta_obs)

# creating the directory to save the results
save_dir = f"Results/posterior_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dictionary to save the posterior samples
post_dict = {}
i = 0
for X in tqdm(X_obs, desc="Generating samples for each X"):
    if (
        task.name == "gaussian_linear_uniform"
        or task.name == "gaussian_linear"
        or task.name == "gaussian_mixture"
        or task.name == "slcp"
        or task.name == "sir"
        or task.name == "lotka_volterra"
    ):
        post_dict[X.reshape(1, -1)] = task._sample_reference_posterior(
            num_samples=n_samples,
            observation=X.reshape(1, -1),
        )
    else:
        post_dict[X.reshape(1, -1)] = task._sample_reference_posterior(
            num_samples=n_samples,
            num_observation=i,
            observation=X.reshape(1, -1),
        )
    i += 1

# Save the posterior samples to a pickle file
with open(os.path.join(save_dir, f"{task.name}_posterior_samples.pkl"), "wb") as f:
    pickle.dump(post_dict, f)
