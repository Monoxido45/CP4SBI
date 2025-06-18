import argparse
import torch
from tqdm import tqdm
import numpy as np
import sbibm
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    "-d",
    help="string for SBI task",
    default="two_moons",
    type=str,
)
parser.add_argument(
    "--seed",
    "-s",
    help="int for random seed to be fixed",
    default=45,
    type=int,
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
# Set the random seed for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# Load the SBI task, simulator, and prior
if task_name != "gaussian_mixture":
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()
    prior = task.get_prior()
else:
    from CP4SBI.gmm_task import GaussianMixture

    task = GaussianMixture(dim=2, prior_bound=3.0)
    simulator = task.get_simulator()
    prior = task.get_prior()


if (
    task_name == "slcp"
    or task_name == "sir"
    or task_name == "lotka_volterra"
    or task_name == "bernoulli_glm"
    or task_name == "bernoulli_glm_raw"
    or task_name == "slcp_distractors"
):
    n_x = 10
else:
    n_x = args.n_x
    theta_obs = prior(num_samples=n_x)
    X_obs = simulator(theta_obs)

    # Print the minimum and maximum of X_obs
    print("Minimum of X_obs:", X_obs.min())
    print("Maximum of X_obs:", X_obs.max())

# creating the directory to save the results
save_dir = f"Results/posterior_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dictionary to save the posterior samples
post_dict = {}
# Check if a checkpoint exists and load it
checkpoint_path = os.path.join(save_dir, f"{task_name}_checkpoint.pkl")
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as checkpoint_file:
        post_dict = pickle.load(checkpoint_file)
    i = len(post_dict)  # Start from the first empty entry
else:
    i = 0

# Generate samples for each X
for j in tqdm(range(i, n_x), desc="Generating samples for each X"):
    if (
        task_name == "gaussian_linear_uniform"
        or task_name == "gaussian_linear"
        or task_name == "gaussian_mixture"
    ):
        X = X_obs[j]
        post_dict[X.reshape(1, -1)] = task._sample_reference_posterior(
            num_samples=n_samples,
            observation=X.reshape(1, -1),
        )
    elif (
        task_name == "slcp"
        or task_name == "sir"
        or task_name == "lotka_volterra"
        or task_name == "bernoulli_glm"
        or task_name == "bernoulli_glm_raw"
        or task_name == "slcp_distractors"
    ):
        X = task.get_observation(num_observation=j + 1)
        post_dict[X] = task.get_reference_posterior_samples(
            num_observation=j + 1,
        )[:n_samples]
    else:
        X = X_obs[i]
        post_dict[X.reshape(1, -1)] = task._sample_reference_posterior(
            num_samples=n_samples,
            num_observation=j + 1,
            observation=X.reshape(1, -1),
        )
    i += 1
    # Save a checkpoint after processing each X
    with open(checkpoint_path, "wb") as checkpoint_file:
        pickle.dump(post_dict, checkpoint_file)

# saving final results
# Delete the checkpoint file before saving the final results
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

# Save the posterior samples to a pickle file
with open(os.path.join(save_dir, f"{task_name}_posterior_samples.pkl"), "wb") as f:
    pickle.dump(post_dict, f)
