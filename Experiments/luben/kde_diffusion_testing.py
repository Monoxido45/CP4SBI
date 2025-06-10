import numpy as np

import pickle
import os
import torch

# testing posterior estimators
from sbi.utils import BoxUniform
from sbi.inference import NPSE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import KDE_HPDScore

# testing naive
from CP4SBI.utils import naive_method
import sbibm
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import gaussian_kde

# for setting input variables
import argparse

original_path = os.getcwd()

# part of the code to debug code
torch.manual_seed(75)
torch.cuda.manual_seed(75)

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
device = "cpu"
prior_NPE = BoxUniform(
    low=-1 * torch.ones(2),
    high=1 * torch.ones(2),
    device=device,
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
X_0 = keys[0]

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


# fitting NPE
inference = NPSE(prior_NPE, device=device)
inference.append_simulations(theta_train, X_train)
inference.train()
posterior = inference.build_posterior()
cuda = device == "cuda"
alpha = 0.1


# A-LOCART applied
locart_conf = BayCon(
    sbi_score=KDE_HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    weighting=True,
    split_calib=False,
    cuda=cuda,
    alpha=alpha,
)

locart_conf.fit(
    X=X_train,
    theta=theta_train,
)

cdf_conf = BayCon(
    sbi_score=KDE_HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="CDF",
    cuda=cuda,
    alpha=alpha,
)

cdf_conf.fit(
    X=X_train,
    theta=theta_train,
)

res = cdf_conf.cdf_split.sbi_score.compute(
    X_calib,
    thetas_calib,
)

locart_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=150,
    using_res=True,
)

cdf_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)


# computing cutoff for observed data
locart_cutoff = locart_conf.predict_cutoff(
    X_obs,
)

cdf_cutoff = cdf_conf.predict_cutoff(
    X_obs,
)

# computing global cutoff
global_cutoff = np.quantile(res, 0.9)

post_estim = deepcopy(locart_conf.locart.sbi_score.posterior)
MAE_locart = np.zeros(len(X_obs))
MAE_global = np.zeros(len(X_obs))
MAE_naive = np.zeros(len(X_obs))
MAE_cdf = np.zeros(len(X_obs))

i = 0
for X_0 in tqdm(keys, desc="Computing coverage across observations"):
    post_samples = X_dict[X_0]

    sample_generated = (
        posterior.sample(
            (1000,),
            x=X_0,
            show_progress_bars=False,
        )
        .cpu()
        .detach()
        .numpy()
    )

    # fitting KDE
    kde = gaussian_kde(sample_generated.T, bw_method="scott")

    # computing log_prob for only one X
    conf_score = -kde(post_samples.T)

    t_cutoff = naive_method(
        posterior,
        X_0,
        alpha=0.1,
        kde=True,
        n_grid=1000,
    )

    coverage_locart = (conf_score <= locart_cutoff[i]).mean()
    coverage_global = (conf_score <= global_cutoff).mean()
    coverage_naive = (conf_score <= t_cutoff).mean()
    coverage_cdf = (conf_score <= cdf_cutoff[i]).mean()

    MAE_locart[i] = np.mean(np.abs(coverage_locart - 0.9))
    MAE_global[i] = np.mean(np.abs(coverage_global - 0.9))
    MAE_naive[i] = np.mean(np.abs(coverage_naive - 0.9))
    MAE_cdf[i] = np.mean(np.abs(coverage_cdf - 0.9))
    i += 1


# Computing mean MAE and standard error for each method
methods = ["locart", "global", "naive", "cdf"]
mae_values = [MAE_locart, MAE_global, MAE_naive, MAE_cdf]

for method, mae in zip(methods, mae_values):
    mean_mae = np.mean(mae)
    std_error = np.std(mae) / np.sqrt(len(mae))
    print(f"{method}: Mean MAE = {mean_mae:.4f}, Standard Error = {std_error:.4f}")
