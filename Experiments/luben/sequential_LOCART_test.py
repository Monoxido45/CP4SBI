import numpy as np

import pickle
import os
import torch

# testing posterior estimators
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore

import sbibm
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import uniform

# for setting input variables
import argparse

original_path = os.getcwd()

# part of the code to debug code
torch.manual_seed(125)
torch.cuda.manual_seed(125)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    "-d",
    help="string for SBI task",
    default="two_moons",
    type=str,
)

if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task

B = 10000
prop_calib = 0.2
B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)
device = "cuda"
prior_NPE = BoxUniform(
    low=-1 * torch.ones(2),
    high=1 * torch.ones(2),
    device=device,
)

if task_name == "two_moons":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(2),
        high=1 * torch.ones(2),
        device=device,
    )

    # Function to evaluate density values for a given 2D point
    def prior_dens(point):
        prior_scipy = uniform(loc=-1, scale=2)
        return prior_scipy.pdf(point[:, 0]) * prior_scipy.pdf(point[:, 1])

elif task_name == "slcp":
    prior_NPE = BoxUniform(
        low=-3 * torch.ones(5),
        high=3 * torch.ones(5),
        device=device,
    )

    # Function to evaluate density values for a given 2D point
    def prior_dens(point):
        prior_scipy = uniform(loc=-3, scale=6)
        return (
            prior_scipy.pdf(point[:, 0])
            * prior_scipy.pdf(point[:, 1])
            * prior_scipy.pdf(point[:, 2])
            * prior_scipy.pdf(point[:, 3])
            * prior_scipy.pdf(point[:, 4])
        )


# getting posterior samples
posterior_data_path = (
    original_path + f"/Results/posterior_data/{task_name}_posterior_samples.pkl"
)
with open(posterior_data_path, "rb") as f:
    X_dict = pickle.load(f)

keys = list(X_dict.keys())
X_obs = torch.cat(list(X_dict.keys())).numpy()

# testing velocity of kde
true_samples = X_dict[keys[0]]

if task_name in ["sclp", "bernoulli_glm"]:
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()
    prior = task.get_prior()
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
    theta_train = theta[train_indices]
    thetas_calib = theta[calib_indices]
else:
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()
    prior = task.get_prior()

    B_train = int(B * (1 - prop_calib))
    B_calib = int(B * prop_calib)
    theta_train = prior(num_samples=B_train)
    X_train = simulator(theta_train)

    # training conformal methods
    thetas_calib = prior(num_samples=B_calib)
    X_calib = simulator(thetas_calib)

# fitting LOCART and A-LOCART to the NPE and computing cutoffs
inference = NPE(prior_NPE, device=device)
inference.append_simulations(theta_train, X_train).train()

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

res = bayes_conf.locart.sbi_score.compute(X_calib, thetas_calib)

# fitting LOCART and A-LOCART to the NPE and computing cutoffs
bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)

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

# computing cutoffs for each X_obs
locart_cutoff = bayes_conf.predict_cutoff(
    X_obs,
)
w_locart_cutoff = w_bayes_conf.predict_cutoff(
    X_obs,
)

MAE_locart = np.zeros(30)
MAE_w_locart = np.zeros(30)
MAE_s_locart = np.zeros(30)
MAE_s_w_locart = np.zeros(30)

i = 0
post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)
if task_name in ["slcp", "bernoulli_glm"]:
    keys_used = keys
    MAE_locart = np.zeros(10)
    MAE_w_locart = np.zeros(10)
    MAE_s_locart = np.zeros(10)
    MAE_s_w_locart = np.zeros(10)
else:
    keys_used = keys[:30]
    MAE_locart = np.zeros(30)
    MAE_w_locart = np.zeros(30)
    MAE_s_locart = np.zeros(30)
    MAE_s_w_locart = np.zeros(30)

for X_0 in tqdm(keys_used, desc="Computing coverage across observations"):
    post_samples = X_dict[X_0]

    # generating samples from the posterior
    theta_t = (
        post_estim.sample(
            (2000,),
            x=X_0.to(device=device),
            show_progress_bars=False,
        )
        .cpu()
        .detach()
    )

    # simulating using theta_t
    X_t = simulator(theta_t)

    res_t = bayes_conf.locart.sbi_score.compute(X_t, theta_t)

    # re-fitting LOCART for the X_0
    print("Re-fitting LOCART for the X_0:", X_0)
    cutoff_obs = bayes_conf.retrain_obs(
        X_0,
        X_new=X_t,
        theta_new=theta_t,
        prior_density_obj=prior_dens,
        min_samples_leaf=300,
        res=res_t,
        using_res=True,
    )

    print("Re-fitting A-LOCART for the X_0:", X_0)
    w_cutoff_obs = w_bayes_conf.retrain_obs(
        X_0,
        X_new=X_t,
        theta_new=theta_t,
        prior_density_obj=prior_dens,
        min_samples_leaf=300,
        res=res_t,
        using_res=True,
    )

    conf_scores = -np.exp(
        post_estim.log_prob(
            post_samples.to(device=device),
            x=X_0.to(device=device),
        )
        .cpu()
        .numpy()
    )

    # computing coverage
    print("Computing coverage for the X_0:", X_0)
    coverage_locart = (conf_scores <= locart_cutoff[i]).mean()
    coverage_w_locart = (conf_scores <= w_locart_cutoff[i]).mean()
    coverage_s_locart = (conf_scores <= cutoff_obs).mean()
    coverage_s_a_locart = (conf_scores <= w_cutoff_obs).mean()

    MAE_locart[i] = np.mean(np.abs(coverage_locart - 0.9))
    MAE_w_locart[i] = np.mean(np.abs(coverage_w_locart - 0.9))
    MAE_s_locart[i] = np.mean(np.abs(coverage_s_locart - 0.9))
    MAE_s_w_locart[i] = np.mean(np.abs(coverage_s_a_locart - 0.9))
    i += 1

# Computing mean MAE and standard error for each method
methods = ["locart", "A-locart", "S-locart", "S-A-locart"]
mae_values = [MAE_locart, MAE_w_locart, MAE_s_locart, MAE_s_w_locart]

for method, mae in zip(methods, mae_values):
    mean_mae = np.mean(mae)
    std_error = np.std(mae) / np.sqrt(len(mae))
    print(f"{method}: Mean MAE = {mean_mae:.4f}, Standard Error = {std_error:.4f}")
