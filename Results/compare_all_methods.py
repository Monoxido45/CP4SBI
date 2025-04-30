# for posterior estimation and calibration
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from sbi.utils.user_input_checks import process_prior

# for benchmarking
import sbibm
from copy import deepcopy
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal

# for plotting and broadcasting
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# for setting input variables
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", "-d", help="string for SBI task", default="two_moons")
parser.add_argument(
    "--seed", "-s", help="int for random seed to be fixed", default=45, type=int
)
parser.add_argument(
    "--B",
    "-B",
    help="int for simulation budget",
    default=1000,
    type=int,
)
parser.add_argument(
    "--prop_calib",
    "-p_calib",
    help="float between 0 and 1 for proportion of calibration data",
    default=0.2,
    type=int,
)

if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task
seed = args.seed
B = args.B
p_calib = args.prop_calib

# Set the random seed for reproducibility
alpha = 0.1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


# Load the SBI task, simulator, and prior
task = sbibm.get_task(task_name)
simulator = task.get_simulator()
prior = task.get_prior()

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

# unused simulators
# elif task.name == "bernoulli_glm":
# setting parameters for prior distribution
#    M = task.dim_parameters - 1
#    D = torch.diag(torch.ones(M)) - torch.diag(torch.ones(M - 1), -1)
#    F = torch.matmul(D, D) + torch.diag(1.0 * torch.arange(M) / (M)) ** 0.5
#    Binv = torch.zeros(size=(M + 1, M + 1))
#    Binv[0, 0] = 0.5  # offset
#    Binv[1:, 1:] = torch.matmul(F.T, F)
# setting up prior distribution using torch
#    prior_params = {"loc": torch.zeros((M + 1,)), "precision_matrix": Binv}
#    prior_dist = MultivariateNormal(**prior_params, validate_args=False)
#    prior_NPE, _, _ = process_prior(prior_dist)
