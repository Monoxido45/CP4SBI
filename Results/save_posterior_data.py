import argparse
import torch
from sbi.utils import BoxUniform
from tqdm import tqdm
import numpy as np
import sbibm
import pandas as pd
import os
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
import math
from sbi.utils.user_input_checks import process_prior

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


# defining the NPE prior according to the chosen task
if task.name == "two_moons":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(2),
        high=1 * torch.ones(2),
    )
elif task.name == "gaussian_linear_uniform":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(10),
        high=1 * torch.ones(10),
    )
elif task.name == "slcp":
    prior_NPE = BoxUniform(
        low=-3 * torch.ones(5),
        high=3 * torch.ones(5),
    )
elif task.name == "bernoulli_glm":
    # setting parameters for prior distribution
    M = task.dim_parameters - 1
    D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
    F = torch.matmul(D, D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
    Binv = torch.zeros(size=(M + 1, M + 1))
    Binv[0, 0] = 0.5  # offset
    Binv[1:, 1:] = torch.matmul(F.T, F)

    # setting up prior distribution using torch
    prior_params = {"loc": torch.zeros((M + 1,)), "precision_matrix": Binv}
    prior_dist = MultivariateNormal(**prior_params, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
elif task.name == "gaussian_linear":
    prior_params = {
        "loc": torch.zeros((task.dim_parameters,)),
        "precision_matrix": torch.inverse(
            task.prior_scale * torch.eye(task.dim_parameters)
        ),
    }
    prior_dist = MultivariateNormal(**prior_params, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
elif task.name == "gaussian_mixture":
    prior_NPE = BoxUniform(
        low=-10 * torch.ones(task.dim_parameters),
        high=10 * torch.ones(task.dim_parameters),
    )
elif task.name == "sir":
    prior_params = {
        "loc": torch.tensor([math.log(0.4), math.log(0.125)]),
        "scale": torch.tensor([0.5, 0.2]),
    }
    prior_dist = LogNormal(**prior_params, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
elif task.name == "lotka_volterra":
    mu_p1 = -0.125
    mu_p2 = -3.0
    sigma_p = 0.5
    prior_params = {
        "loc": torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2]),
        "scale": torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p]),
    }
    prior_dist = LogNormal(**prior_params, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)


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
    if task.name == "two_moons":
        post_dict[X.reshape(1, -1)] = task._sample_reference_posterior(
            num_samples=n_samples,
            num_observation=i,
            observation=X.reshape(1, -1),
        )
    i += 1

# Save the posterior samples to a pickle file
with open(os.path.join(save_dir, f"{task.name}_posterior_samples.pkl"), "wb") as f:
    pickle.dump(post_dict, f)
