import numpy as np
import pickle
import os
import torch

from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from CP4SBI.utils import naive_method, hdr_method
from tqdm import tqdm
import sbibm
from copy import deepcopy

from torch.distributions.multivariate_normal import MultivariateNormal
from sbi.utils.user_input_checks import process_prior

# for setting input variables
import argparse

original_path = os.getcwd()
# Set random seeds for reproducibility
torch.manual_seed(45)
torch.cuda.manual_seed(45)
alpha = 0.1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    "-d",
    help="string for SBI task",
    default="gaussian_linear_uniform",
    type=str,
)
parser.add_argument(
    "--device",
    "-dvc",
    help="string for device to be used",
    default="cuda",
    type=str,
)

if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task
device = args.device

B = 10000
prop_calib = 0.2
B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)


if task_name == "gaussian_linear_uniform":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(2),
        high=1 * torch.ones(2),
        device=device,
    )
elif task_name == "slcp":
    prior_NPE = BoxUniform(
        low=-3 * torch.ones(2),
        high=3 * torch.ones(2),
        device=device,
    )
elif task_name == "gaussian_linear":
    prior_params = {
        "loc": torch.zeros((2,), device=device),
        "precision_matrix": torch.inverse(0.1 * torch.eye(2, device=device)),
    }
    prior_dist = MultivariateNormal(
        **prior_params,
        validate_args=False,
    )
    prior_NPE, _, _ = process_prior(prior_dist)
elif task_name == "bernoulli_glm" or "bernoulli_glm_raw":
    dim_parameters = 2
    # parameters for the prior distribution
    M = dim_parameters - 1
    D = torch.diag(torch.ones(M, device=device)) - torch.diag(
        torch.ones(M - 1, device=device), -1
    )
    F = (
        torch.matmul(D, D)
        + torch.diag(1.0 * torch.arange(M, device=device) / (M)) ** 0.5
    )
    Binv = torch.zeros(size=(M + 1, M + 1), device=device)
    Binv[0, 0] = 0.5  # offset
    Binv[1:, 1:] = torch.matmul(F.T, F)  # filter

    prior_params = {
        "loc": torch.zeros((M + 1,), device=device),
        "precision_matrix": Binv,
    }

    prior_dist = MultivariateNormal(
        **prior_params,
        validate_args=False,
    )
    prior_NPE, _, _ = process_prior(prior_dist)


# Data split parameters
B = 10000
prop_calib = 0.2


if task_name in ["sclp", "sir", "lotka_volterra", "bernoulli_glm"]:
    # Load the X_list pickle file from the X_data folder
    x_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_X_samples_10000.pkl"
    )
    with open(x_data_path, "rb") as f:
        X_list = pickle.load(f)

    # Load the X_list pickle file from the X_data folder
    theta_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_theta_samples_10000.pkl"
    )
    with open(theta_data_path, "rb") as f:
        theta_list = pickle.load(f)

    # exploring only the first element
    X = X_list[0]
    theta = theta_list[0]

    # splitting X
    indices = torch.randperm(X.shape[0])
    train_indices = indices[:B_train]
    calib_indices = indices[B_train:]

    X_train = X[train_indices]
    X_calib = X[calib_indices]

    # splitting theta
    theta_train = theta[train_indices, :2]
    thetas_calib = theta[calib_indices, :2]

else:
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()
    prior = task.get_prior()

    B_train = int(B * (1 - prop_calib))
    B_calib = int(B * prop_calib)

    # Generate training and calibration data
    theta_train_all = prior(num_samples=B_train)
    X_train = simulator(theta_train_all)
    theta_train = theta_train_all[:, :2]

    thetas_calib_all = prior(num_samples=B_calib)
    X_calib = simulator(thetas_calib_all)
    thetas_calib = thetas_calib_all[:, :2]

# Fit NPE
inference = NPE(prior_NPE, device=device)
inference.append_simulations(theta_train, X_train).train()

# Fit all calibration methods
cuda = device == "cuda"

# Fit BayCon (LOCART) calibration method
bayes_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    split_calib=False,
    cuda=device == "cuda",
    alpha=0.1,
)
bayes_conf.fit(
    X=X_train,
    theta=theta_train,
)

# Compute conformity scores for calibration data
res = bayes_conf.locart.sbi_score.compute(X_calib, thetas_calib)

# Calibrate LOCART
bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)

# Optionally, fit weighted LOCART (A-LOCART)
w_bayes_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    split_calib=False,
    cuda=device == "cuda",
    alpha=0.1,
    weighting=True,
)
w_bayes_conf.fit(
    X=X_train,
    theta=theta_train,
)
w_bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)


# Global conformal
global_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="global",
    cuda=cuda,
    alpha=alpha,
)
global_conf.fit(X=X_train, theta=theta_train)
global_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)

# CDF conformal
cdf_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="CDF",
    cuda=cuda,
    alpha=alpha,
)
cdf_conf.fit(X=X_train, theta=theta_train)
cdf_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)

# Local CDF conformal
local_cdf_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="CDF local",
    split_calib=False,
    cuda=cuda,
    alpha=alpha,
)
local_cdf_conf.fit(X=X_train, theta=theta_train)
local_cdf_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)

# Import posterior samples dictionary
original_path = os.getcwd()
posterior_data_path = os.path.join(
    original_path, f"Results/posterior_data/{task_name}_posterior_samples.pkl"
)
with open(posterior_data_path, "rb") as f:
    X_dict = pickle.load(f)

X_obs = torch.cat(list(X_dict.keys())).numpy()

# HDR recalibration (using HPDScore)
hdr_cutoff, hdr_obj = hdr_method(
    post_estim=inference,
    X_calib=X_calib,
    thetas_calib=thetas_calib,
    n_grid=1000,
    X_test=X_obs,
    is_fitted=True,
    alpha=alpha,
    score_type="HPD",
    device=device,
    kde=False,
)

# Compute cutoffs for X_obs for each method using predict_cutoff
locart_cutoff = bayes_conf.predict_cutoff(X_obs)
w_locart_cutoff = w_bayes_conf.predict_cutoff(X_obs)
global_cutoff = global_conf.predict_cutoff(X_obs)
cdf_cutoff = cdf_conf.predict_cutoff(X_obs)
local_cdf_cutoff = local_cdf_conf.predict_cutoff(X_obs)

# Compute coverage for each X in X_dict for all methods
coverage_locart = []
coverage_w_locart = []
coverage_global = []
coverage_cdf = []
coverage_local_cdf = []
coverage_hdr = []
coverage_naive = []

# Compute naive cutoff for each X in X_obs
naive_cutoff = []
for i, X_0 in enumerate(tqdm(list(X_dict.keys()), desc="Coverage for each X")):
    post_samples = X_dict[X_0][:, :2]
    post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)

    # Compute conformity scores for posterior samples
    conf_scores = -np.exp(
        post_estim.log_prob(
            post_samples.to(device=device),
            x=X_0.to(device=device),
        )
        .cpu()
        .numpy()
    )

    # LOCART
    coverage_locart.append(np.mean(conf_scores <= locart_cutoff[i]))
    # A-LOCART
    coverage_w_locart.append(np.mean(conf_scores <= w_locart_cutoff[i]))
    # Global
    coverage_global.append(np.mean(conf_scores <= global_cutoff[i]))
    # CDF
    coverage_cdf.append(np.mean(conf_scores <= cdf_cutoff[i]))
    # Local CDF
    coverage_local_cdf.append(np.mean(conf_scores <= local_cdf_cutoff[i]))
    # HDR
    _, dens_samples = hdr_obj.recal_sample(
        y_hat=post_samples.reshape(1, post_samples.shape[0], post_samples.shape[1]),
        f_hat_y_hat=-conf_scores.reshape(1, -1),
    )
    hdr_conf_scores = -dens_samples[0, :]
    coverage_hdr.append(np.mean(hdr_conf_scores <= hdr_cutoff[i]))
    # Naive
    naive_c = naive_method(
        post_estim,
        X=X_0,
        alpha=alpha,
        score_type="HPD",
        device=device,
        B_naive=1000,
    )
    naive_cutoff.append(naive_c)
    coverage_naive.append(np.mean(conf_scores <= naive_c))

# Convert to numpy arrays for further analysis
coverage_locart = np.array(coverage_locart)
coverage_w_locart = np.array(coverage_w_locart)
coverage_global = np.array(coverage_global)
coverage_cdf = np.array(coverage_cdf)
coverage_local_cdf = np.array(coverage_local_cdf)
coverage_hdr = np.array(coverage_hdr)
coverage_naive = np.array(coverage_naive)
naive_cutoff = np.array(naive_cutoff)


# Print mean absolute deviation from nominal coverage for each method
nominal = 1 - alpha
print(f"LOCART MAD: {np.mean(np.abs(coverage_locart - nominal)):.4f}")
print(f"A-LOCART MAD: {np.mean(np.abs(coverage_w_locart - nominal)):.4f}")
print(f"Global MAD: {np.mean(np.abs(coverage_global - nominal)):.4f}")
print(f"CDF MAD: {np.mean(np.abs(coverage_cdf - nominal)):.4f}")
print(f"Local CDF MAD: {np.mean(np.abs(coverage_local_cdf - nominal)):.4f}")
print(f"HDR MAD: {np.mean(np.abs(coverage_hdr - nominal)):.4f}")
print(f"Naive MAD: {np.mean(np.abs(coverage_naive - nominal)):.4f}")
