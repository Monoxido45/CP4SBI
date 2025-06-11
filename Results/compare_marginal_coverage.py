# for posterior estimation and calibration
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE, NPSE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore, WALDOScore
from sbi.utils.user_input_checks import process_prior
from sbi.utils import MultipleIndependent
from CP4SBI.utils import naive_method, hdr_method

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
import pickle
import os

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
    "--score",
    "-sc",
    help="string for score to be used",
    default="HPD",
    type=str,
)

parser.add_argument(
    "--device",
    "-dvc",
    help="string for device to be used",
    default="cuda",
    type=str,
)
parser.add_argument(
    "--n_rep",
    "-n_rep",
    help="int for number of repetitions",
    default=30,
    type=int,
)

parser.add_argument(
    "--X_list",
    "-X_list",
    help="string indicating whether to use X_list or not",
    default="False",
    type=str,
)

parser.add_argument(
    "--X_test_list",
    "-X_list_t",
    help="string indicating whether to use X_test_list or not",
    default="False",
    type=str,
)
parser.add_argument(
    "--B",
    "-B",
    help="int for simulation budget",
    default=10000,
    type=int,
)
parser.add_argument(
    "--B_t",
    "-B_t",
    help="int for simulation budget for test set",
    default=2000,
    type=int,
)
parser.add_argument(
    "--prop_calib",
    "-p_calib",
    help="float between 0 and 1 for proportion of calibration data",
    default=0.2,
    type=int,
)
parser.add_argument(
    "--sample_with",
    "-sw",
    help="string for sampling method to be used inside direct_posterior",
    default="direct",
    type=str,
)

parser.add_argument(
    "--base_model",
    "-b_m",
    help="string for base model to be used",
    default="NPE",
    type=str,
)

original_path = os.getcwd()
if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task
seed = args.seed
B = args.B
B_test = args.B_t
p_calib = args.prop_calib
n_rep = args.n_rep
device = args.device
score_type = args.score
X_str = args.X_list == "True"
X_str_t = args.X_test_list == "True"
sample_with = args.sample_with
base_model = args.base_model

if X_str:
    # Load the X_list pickle file from the X_data folder
    x_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_X_samples_{B}.pkl"
    )
    with open(x_data_path, "rb") as f:
        X_data = pickle.load(f)

    # Load the X_list pickle file from the X_data folder
    theta_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_theta_samples_{B}.pkl"
    )
    with open(theta_data_path, "rb") as f:
        theta_list = pickle.load(f)

    X_list = {"X": X_data, "theta": theta_list}
else:
    X_list = None


if X_str_t:
    # Load the X_test_list pickle file from the X_data folder
    x_test_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_X_test_samples_{B_test}.pkl"
    )
    with open(x_test_data_path, "rb") as f:
        X_test_data = pickle.load(f)

    # Load the theta_test_list pickle file from the X_data folder
    theta_test_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_theta_test_samples_{B_test}.pkl"
    )
    with open(theta_test_data_path, "rb") as f:
        theta_test_list = pickle.load(f)

    X_test_list = {"X_test": X_test_data, "theta_test": theta_test_list}
else:
    X_test_list = None

# Set the random seed for reproducibility
alpha = 0.1

# Load the SBI task, simulator, and prior
if task_name != "gaussian_mixture":
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()
    prior = task.get_prior()
else:
    from CP4SBI.gmm_task import GaussianMixture

    task = GaussianMixture(dim=2, prior_bound=4.0)
    simulator = task.get_simulator()
    prior = task.get_prior()


if task_name == "two_moons":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(2),
        high=1 * torch.ones(2),
        device=device,
    )
elif task_name == "gaussian_linear_uniform":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(10),
        high=1 * torch.ones(10),
        device=device,
    )
elif task_name == "slcp":
    prior_NPE = BoxUniform(
        low=-3 * torch.ones(5),
        high=3 * torch.ones(5),
        device=device,
    )
elif task_name == "gaussian_linear":
    prior_params = {
        "loc": torch.zeros((task.dim_parameters,), device=device),
        "precision_matrix": torch.inverse(
            0.1 * torch.eye(task.dim_parameters, device=device)
        ),
    }
    prior_dist = MultivariateNormal(
        **prior_params,
        validate_args=False,
    )
    prior_NPE, _, _ = process_prior(prior_dist)
elif task_name == "bernoulli_glm" or "bernoulli_glm_raw":
    dim_parameters = 10
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
elif task_name == "gaussian_mixture":
    prior_NPE = BoxUniform(
        low=-4 * torch.ones(2),
        high=4 * torch.ones(2),
        device=device,
    )

elif task_name == "sir":
    prior_list = [
        LogNormal(
            loc=torch.tensor([math.log(0.4)], device=device),
            scale=torch.tensor([0.5], device=device),
            validate_args=False,
        ),
        LogNormal(
            loc=torch.tensor([math.log(0.125)], device=device),
            scale=torch.tensor([0.2], device=device),
            validate_args=False,
        ),
    ]
    prior_dist = MultipleIndependent(prior_list, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
elif task_name == "lotka_volterra":
    mu_p1 = -0.125
    mu_p2 = -3.0
    sigma_p = 0.5
    prior_params = {
        "loc": torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2], device=device),
        "scale": torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p], device=device),
    }

    prior_list = [
        LogNormal(
            loc=torch.tensor([mu_p1], device=device),
            scale=torch.tensor([sigma_p], device=device),
            validate_args=False,
        ),
        LogNormal(
            loc=torch.tensor([mu_p2], device=device),
            scale=torch.tensor([sigma_p], device=device),
            validate_args=False,
        ),
        LogNormal(
            loc=torch.tensor([mu_p1], device=device),
            scale=torch.tensor([sigma_p], device=device),
            validate_args=False,
        ),
        LogNormal(
            loc=torch.tensor([mu_p2], device=device),
            scale=torch.tensor([sigma_p], device=device),
            validate_args=False,
        ),
    ]
    prior_dist = MultipleIndependent(prior_list, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)


def compute_coverage(
    prior_NPE,
    score_type,
    split_calib=False,
    X=None,
    theta=None,
    X_test=None,
    theta_test=None,
    B=5000,
    prop_calib=0.2,
    alpha=0.1,
    task_name="two_moons",
    device="cuda",
    random_seed=0,
    min_samples_leaf=300,
    naive_samples=1000,
    sample_with="direct",
):
    # setting seet
    if not task_name == "gaussian_mixture":
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    # checking if X_list is None or not
    # splitting simulation budget
    B_train = int(B * (1 - prop_calib))
    B_calib = int(B * prop_calib)

    if X is None and theta is None:
        # training samples
        theta_train = prior(num_samples=B_train)
        X_train = simulator(theta_train)

        # training conformal methods
        thetas_calib = prior(num_samples=B_calib)
        X_calib = simulator(thetas_calib)

    else:
        # splitting X
        indices = torch.randperm(X.shape[0])
        train_indices = indices[:B_train]
        calib_indices = indices[B_train:]

        X_train = X[train_indices]
        X_calib = X[calib_indices]

        # splitting theta
        theta_train = theta[train_indices]
        thetas_calib = theta[calib_indices]

    if X_test is None and theta_test is None:
        # training conformal methods
        thetas_test = prior(num_samples=B_test)
        X_test = simulator(thetas_test)
    else:
        thetas_test = theta_test
        X_test = X_test

    if base_model == "NPE":
        # fitting NPE
        inference = NPE(prior_NPE, device=device)
        inference.append_simulations(
            theta_train,
            X_train,
        ).train()

    elif base_model == "NPSE":
        # fitting diffusion model
        inference = NPSE(prior_NPE, device=device)
        inference.append_simulations(
            theta=theta_train.to(device),
            x=X_train.to(device),
        ).train()

    cuda = device == "cuda"

    # checking score type
    if score_type == "HPD":
        score_used = HPDScore
    elif score_type == "WALDO":
        score_used = WALDOScore

    print("Computing conformal scores")
    cdf_conf = BayCon(
        sbi_score=score_used,
        base_inference=inference,
        is_fitted=True,
        conformal_method="CDF",
        cuda=cuda,
        alpha=alpha,
    )

    cdf_conf.fit(
        X=X_train,
        theta=theta_train,
        sample_with=sample_with,
    )

    res = cdf_conf.cdf_split.sbi_score.compute(X_calib, thetas_calib)

    # CDF split
    print("Fitting CDF split")

    cdf_conf.calib(
        X_calib=X_calib,
        theta_calib=res,
        using_res=True,
    )

    print("Fitting local CDF split")
    # CDF split + LOCART
    local_cdf_conf = BayCon(
        sbi_score=score_used,
        base_inference=inference,
        is_fitted=True,
        conformal_method="CDF local",
        split_calib=split_calib,
        cuda=cuda,
        alpha=alpha,
    )

    local_cdf_conf.fit(
        X=X_train,
        theta=theta_train,
        sample_with=sample_with,
    )

    local_cdf_conf.calib(
        X_calib=X_calib,
        theta_calib=res,
        min_samples_leaf=min_samples_leaf,
        using_res=True,
    )

    # fitting LOCART
    print("Fitting LOCART")
    bayes_conf = BayCon(
        sbi_score=score_used,
        base_inference=inference,
        is_fitted=True,
        conformal_method="local",
        split_calib=split_calib,
        cuda=cuda,
        alpha=alpha,
    )
    bayes_conf.fit(
        X=X_train,
        theta=theta_train,
    )
    bayes_conf.calib(
        X_calib=X_calib,
        theta_calib=res,
        min_samples_leaf=min_samples_leaf,
        using_res=True,
    )

    # fitting LOCART
    print("Fitting A-LOCART")
    w_bayes_conf = BayCon(
        sbi_score=score_used,
        base_inference=inference,
        is_fitted=True,
        conformal_method="local",
        weighting=True,
        split_calib=split_calib,
        cuda=cuda,
        alpha=alpha,
    )
    w_bayes_conf.fit(
        X=X_train,
        theta=theta_train,
        sample_with=sample_with,
    )
    w_bayes_conf.calib(
        X_calib=X_calib,
        theta_calib=res,
        min_samples_leaf=min_samples_leaf,
        using_res=True,
    )

    # global
    print("Fitting global conformal")
    global_conf = BayCon(
        sbi_score=score_used,
        base_inference=inference,
        is_fitted=True,
        conformal_method="global",
        cuda=cuda,
        alpha=alpha,
    )

    global_conf.fit(
        X=X_train,
        theta=theta_train,
    )

    global_conf.calib(
        X_calib=X_calib,
        theta_calib=res,
        using_res=True,
    )

    post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)

    # HDR recalibration
    print("Fitting HDR recalibration")
    hdr_cutoff, hdr_obj = hdr_method(
        post_estim=inference,
        X_calib=X_calib,
        thetas_calib=thetas_calib,
        n_grid=1000,
        X_test=X_test,
        is_fitted=True,
        alpha=alpha,
        score_type=score_type,
        device=device,
    )

    locart_cutoff = bayes_conf.predict_cutoff(X_test)
    global_cutoff = global_conf.predict_cutoff(X_test)
    cdf_cutoff = cdf_conf.predict_cutoff(X_test)
    local_cdf_cutoff = local_cdf_conf.predict_cutoff(X_test)
    alocart_cutoff = w_bayes_conf.predict_cutoff(X_test)

    coverage_locart = np.zeros(X_test.shape[0])
    coverage_global = np.zeros(X_test.shape[0])
    coverage_naive = np.zeros(X_test.shape[0])
    coverage_cdf = np.zeros(X_test.shape[0])
    coverage_local_cdf = np.zeros(X_test.shape[0])
    coverage_a_locart = np.zeros(X_test.shape[0])
    coverage_hdr = np.zeros(X_test.shape[0])

    # computing conf scores for each X_0 and theta_0
    conf_scores = cdf_conf.cdf_split.sbi_score.compute(X_test, thetas_test)

    i = 0
    # computing cutoffs for naive
    naive_cutoff = np.zeros(X_test.shape[0])
    for X_0, theta_0 in tqdm(
        zip(X_test, thetas_test), desc="Computing naive cutoff for each test set"
    ):
        if len(X_0.shape) == 1:
            X_0 = X_0.reshape(1, -1)
        if len(theta_0.shape) == 1:
            theta_0 = theta_0.reshape(1, -1)

        # sampling for compute marginal coverage for hdr
        theta_s = post_estim.sample(
            (1000,),
            x=X_0.to(device=device),
            show_progress_bars=False,
        )

        new_conf_scores = -np.exp(
            post_estim.log_prob(
                theta_s.to(device=device),
                x=X_0.to(device=device),
            )
            .cpu()
            .numpy()
        )

        # recalibrating sample
        _, dens_samples = hdr_obj.recal_sample(
            y_hat=theta_s.cpu().reshape(
                1,
                theta_s.shape[0],
                theta_s.shape[1],
            ),
            f_hat_y_hat=-new_conf_scores.reshape(1, -1),
        )

        hdr_conf_scores = -dens_samples[0, :]
        coverage_hdr[i] = np.mean(hdr_conf_scores <= hdr_cutoff[i])
        if score_type == "HPD":
            # computing naive cutoff
            if (
                task_name == "sir"
                or task_name == "lotka_volterra"
                or task_name == "gaussian_linear"
            ):
                closest_t = naive_method(
                    post_estim,
                    X=X_0,
                    alpha=alpha,
                    score_type=score_type,
                    device=device,
                    n_grid=1000,
                    B_naive=naive_samples,
                )
            else:
                closest_t = naive_method(
                    post_estim,
                    X=X_0,
                    alpha=alpha,
                    score_type=score_type,
                    device=device,
                    B_naive=naive_samples,
                )
            naive_cutoff[i] = closest_t

        # computing coverage
        coverage_locart[i] = (conf_scores[i] <= locart_cutoff[i]) + 0
        coverage_global[i] = (conf_scores[i] <= global_cutoff[i]) + 0
        coverage_naive[i] = (conf_scores[i] <= naive_cutoff[i]) + 0
        coverage_cdf[i] = (conf_scores[i] <= cdf_cutoff[i]) + 0
        coverage_local_cdf[i] = (conf_scores[i] <= local_cdf_cutoff[i]) + 0
        coverage_a_locart[i] = (conf_scores[i] <= alocart_cutoff[i]) + 0
        i += 1

    # Creating a pandas DataFrame with the mean coverage values
    coverage_df = pd.DataFrame(
        {
            "LOCART MAD": [np.mean(coverage_locart)],
            "A-LOCART MAD": [np.mean(coverage_a_locart)],
            "Global CP MAD": [np.mean(coverage_global)],
            "Naive MAD": [np.mean(coverage_naive)],
            "CDF MAD": [np.mean(coverage_cdf)],
            "Local CDF MAD": [np.mean(coverage_local_cdf)],
            "HDR MAD": [np.mean(coverage_hdr)],
        }
    )

    return coverage_df


def compute_coverage_repeated(
    prior_NPE,
    score_type,
    X_list=None,
    X_test_list=None,
    B=5000,
    prop_calib=0.2,
    alpha=0.1,
    task_name="two_moons",
    device="cuda",
    central_seed=0,
    min_samples_leaf=150,
    naive_samples=1000,
    n_rep=30,
    sample_with="direct",
):
    # Generate an array of seeds using the central_seed
    seeds = np.random.RandomState(central_seed).randint(0, 2**32 - 1, size=n_rep)

    # Initialize an empty list to store coverage results
    coverage_results = []

    # Initialize a list to store checkpoints
    checkpoint_path = os.path.join(original_path, "Results", "Marginal_results")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(
        checkpoint_path, f"{score_type}_{task_name}_checkpoints.pkl"
    )
    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            coverage_results = pickle.load(f)

    # Start the loop from the length of the checkpoint list
    start_index = len(coverage_results)
    if task_name == "gaussian_mixture":
        torch.manual_seed(central_seed)
        torch.cuda.manual_seed(central_seed)

    # Adjust the loop to start from the start_index
    for j, seed in enumerate(
        tqdm(seeds[start_index:], desc="Computing coverage for each seed"),
        start=start_index,
    ):
        # checking X_list
        if X_list is not None:
            X = X_list["X"][j]
            theta = X_list["theta"][j]
        else:
            X = None
            theta = None

        if X_test_list is not None:
            X_test = X_test_list["X_test"][j]
            theta_test = X_test_list["theta_test"][j]
        else:
            X_test = None
            theta_test = None

        coverage_df = compute_coverage(
            score_type=score_type,
            prior_NPE=prior_NPE,
            X=X,
            theta=theta,
            X_test=X_test,
            theta_test=theta_test,
            B=B,
            prop_calib=prop_calib,
            alpha=alpha,
            task_name=task_name,
            device=device,
            random_seed=seed,
            min_samples_leaf=min_samples_leaf,
            naive_samples=naive_samples,
            sample_with=sample_with,
        )
        coverage_results.append(coverage_df)

        with open(checkpoint_file, "wb") as f:
            pickle.dump(coverage_results, f)

    # Combine results into a single DataFrame
    combined_coverage_df = pd.concat(coverage_results, ignore_index=True)
    return combined_coverage_df


all_coverage_df = compute_coverage_repeated(
    score_type=score_type,
    prior_NPE=prior_NPE,
    X_list=X_list,
    X_test_list=X_test_list,
    B=B,
    prop_calib=p_calib,
    alpha=alpha,
    task_name=task_name,
    device=device,
    central_seed=seed,
    min_samples_leaf=300,
    naive_samples=1000,
    n_rep=n_rep,
    sample_with=sample_with,
)

# Create the "MAE_results" folder if it doesn't exist for NPE
mae_results_path = os.path.join(original_path, "Results", "Marginal_results")
os.makedirs(mae_results_path, exist_ok=True)


# Save the all_coverage_df DataFrame to a CSV file
csv_path = os.path.join(
    mae_results_path, f"{score_type}_{task_name}_coverage_results_{B}.csv"
)
all_coverage_df.to_csv(csv_path, index=False)

# Compute the summary statistics (mean and standard error) for each column
summary_stats = all_coverage_df.agg(["mean", "std"])
summary_stats.loc["stderr"] = summary_stats.loc["std"] / np.sqrt(n_rep)
summary_stats = summary_stats.drop(index="std")  # Drop standard deviation row

# Save the summary statistics to a CSV file
summary_csv_path = os.path.join(
    mae_results_path, f"{score_type}_{task_name}_coverage_summary_{B}.csv"
)
summary_stats.to_csv(summary_csv_path)

# Removing all checkpoints
checkpoint_file = os.path.join(
    mae_results_path, f"{score_type}_{task_name}_checkpoints.pkl"
)
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
