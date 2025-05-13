# for posterior estimation and calibration
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore, WALDOScore
from sbi.utils.user_input_checks import process_prior
from sbi.utils import MultipleIndependent
from CP4SBI.utils import naive_method

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
    "--B",
    "-B",
    help="int for simulation budget",
    default=10000,
    type=int,
)

parser.add_argument(
    "--prop_calib",
    "-p_calib",
    help="float between 0 and 1 for proportion of calibration data",
    default=0.2,
    type=int,
)


original_path = os.getcwd()
if __name__ == "__main__":
    args = parser.parse_args()  # get arguments from command line
else:
    args = parser.parse_args("")  # get default arguments

task_name = args.task
seed = args.seed
B = args.B
p_calib = args.prop_calib
n_rep = args.n_rep
device = args.device
score_type = args.score
X_str = (args.X_list == "True")

if X_str:
    # Load the X_list pickle file from the X_data folder
    x_data_path = os.path.join(original_path, 
                               "Results/X_data", 
                               f"{task_name}_X_samples.pkl")
    with open(x_data_path, "rb") as f:
        X_list = pickle.load(f)
    
    # Load the X_list pickle file from the X_data folder
    theta_data_path = os.path.join(original_path, 
                               "Results/X_data", 
                               f"{task_name}_theta_samples.pkl")
    with open(theta_data_path, "rb") as f:
        theta_list = pickle.load(f)
    
    X_dict = {"X": X_list, "theta": theta_list}
else:
    X_dict = None

# Set the random seed for reproducibility
alpha = 0.1

# Load the SBI task, simulator, and prior
task = sbibm.get_task(task_name)
simulator = task.get_simulator()
prior = task.get_prior()

if task.name == "two_moons":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(2),
        high=1 * torch.ones(2),
        device=device,
    )
elif task.name == "gaussian_linear_uniform":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(10),
        high=1 * torch.ones(10),
        device=device,
    )
elif task.name == "slcp":
    prior_NPE = BoxUniform(
        low=-3 * torch.ones(5),
        high=3 * torch.ones(5),
        device=device,
    )
elif task.name == "gaussian_linear":
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
elif task.name == "gaussian_mixture":
    prior_NPE = BoxUniform(
        low=-10 * torch.ones(task.dim_parameters),
        high=10 * torch.ones(task.dim_parameters),
        device=device,
    )
elif task.name == "sir":
    prior_list = [
        LogNormal(loc = torch.tensor([math.log(0.4)], device = device),
                  scale = torch.tensor([0.5], device = device),
                  validate_args = False,
                  ),
        LogNormal(loc = torch.tensor([math.log(0.125)], device = device),
                  scale = torch.tensor([0.2], device = device),
                  validate_args = False,
                  ),   
    ]
    prior_dist = MultipleIndependent(prior_list, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
elif task.name == "lotka_volterra":
    mu_p1 = -0.125
    mu_p2 = -3.0
    sigma_p = 0.5
    prior_params = {
        "loc": torch.tensor([mu_p1, mu_p2, mu_p1, mu_p2], device=device),
        "scale": torch.tensor([sigma_p, sigma_p, sigma_p, sigma_p], device=device),
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


def compute_coverage(
    prior_NPE,
    score_type,
    X = None,
    theta = None,
    B=5000,
    prop_calib=0.2,
    alpha=0.1,
    num_obs=500,
    task_name="two_moons",
    device="cuda",
    random_seed=0,
    min_samples_leaf=300,
    naive_samples=1000,
):
    # fixing task
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # setting seet
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

        # fitting NPE
        inference = NPE(prior_NPE, device=device)
        inference.append_simulations(theta_train, X_train).train()

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

         # fitting NPE
        inference = NPE(prior_NPE, device=device)
        inference.append_simulations(theta_train, 
                                     X_train).train()

    cuda = device == "cuda"

    # checking score type
    if score_type == "HPD":
        score_used = HPDScore
    elif score_type == "WALDO":
        score_used = WALDOScore

    # CDF split
    print("Fitting CDF split")
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
    )

    cdf_conf.calib(
        X_calib=X_calib,
        theta_calib=thetas_calib,
    )

    # fitting LOCART
    print("Fitting LOCART")
    bayes_conf = BayCon(
        sbi_score=score_used,
        base_inference=inference,
        is_fitted=True,
        conformal_method="local",
        cuda=cuda,
        alpha=alpha,
    )
    bayes_conf.fit(
        X=X_train,
        theta=theta_train,
    )

    bayes_conf.calib(
        X_calib=X_calib,
        theta_calib=thetas_calib,
        locart_kwargs={"min_samples_leaf": min_samples_leaf},
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
        theta_calib=thetas_calib,
    )

    coverage_locart = np.zeros(num_obs)
    coverage_global = np.zeros(num_obs)
    coverage_cdf = np.zeros(num_obs)
    coverage_naive = np.zeros(num_obs)

    # Load the dictionary from the pickle file
    posterior_data_path = (
        original_path + f"/Results/posterior_data/{task_name}_posterior_samples.pkl"
    )
    with open(posterior_data_path, "rb") as f:
        X_dict = pickle.load(f)

    X_obs = torch.cat(list(X_dict.keys())).numpy()
    locart_cutoff = bayes_conf.predict_cutoff(X_obs)
    global_cutoff = global_conf.predict_cutoff(X_obs)
    cdf_cutoff = cdf_conf.predict_cutoff(X_obs)

    i = 0
    dict_keys = list(X_dict.keys())
    # evaluating cutoff for each observation
    for X in tqdm(dict_keys, desc="Computing coverage across observations"):
        post_samples = X_dict[X]

        # computing naive cutoff
        post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)

        if score_type == "HPD":
            # computing naive cutoff
            if task_name == "sir":
                closest_t = naive_method(
                post_estim,
                X=X,
                alpha = alpha,
                score_type=score_type,
                device=device,
                n_grid = 1000,
                B_naive = naive_samples,
                )
            else:
                closest_t = naive_method(
                post_estim,
                X=X,
                alpha = alpha,
                score_type=score_type,
                device=device,
                B_naive = naive_samples,
                )

            # computing scores
            conf_scores = -np.exp(
            post_estim.log_prob_batched(
                post_samples.to(device=device),
                x=X.reshape(1, -1).to(device=device),
            )
            .cpu()
            .numpy()
            )
        elif score_type == "WALDO":
            # computing naive cutoff
            if task_name == "sir":
                closest_t, mean_array, inv_matrix = naive_method(
                X=X,
                alpha = alpha,
                score_type=score_type,
                device=device,
                B_naive = naive_samples,
                n_grid = 700,
                )
            else:
                closest_t, mean_array, inv_matrix = naive_method(
                post_estim,
                X=X,
                alpha = alpha,
                score_type=score_type,
                device=device,
                B_naive = naive_samples,
                )

            # computing scores
            conf_scores = np.zeros(post_samples.shape[0])
            for j in range(post_samples.shape[0]):
                if mean_array.shape[0] > 1:
                    sel_sample = post_samples[j, :].cpu().numpy()
                    conf_scores[j] = (
                    (mean_array - sel_sample).transpose()
                    @ inv_matrix
                    @ (mean_array - sel_sample)
                    )
                else:
                    sel_sample = post_samples[j].cpu().numpy()
                    conf_scores[j] = (
                    (mean_array - sel_sample) ** 2 / (inv_matrix)
                    )

        # computing coverage
        coverage_locart[i] = np.mean(conf_scores <= locart_cutoff[i])
        coverage_global[i] = np.mean(conf_scores <= global_cutoff[i])
        coverage_naive[i] = np.mean(conf_scores <= closest_t)
        coverage_cdf[i] = np.mean(conf_scores <= cdf_cutoff[i])

        i += 1

    # Creating a pandas DataFrame with the mean coverage values
    coverage_df = pd.DataFrame(
        {
            "LOCART MAD": [np.mean(np.abs(coverage_locart - (1 - alpha)))],
            "Global CP MAD": [np.mean(np.abs(coverage_global - (1 - alpha)))],
            "Naive MAD": [np.mean(np.abs(coverage_naive - (1 - alpha)))],
            "CDF MAD": [np.mean(np.abs(coverage_cdf - (1 - alpha)))],
        }
    )
    return coverage_df


def compute_coverage_repeated(
    prior_NPE,
    score_type,
    X_list = None,
    B=5000,
    prop_calib=0.2,
    alpha=0.1,
    num_obs=500,
    task_name="two_moons",
    device="cuda",
    central_seed=0,
    min_samples_leaf=150,
    naive_samples=1000,
    n_rep=30,
):
    # Generate an array of seeds using the central_seed
    seeds = np.random.RandomState(central_seed).randint(0, 2**32 - 1, size=n_rep)

    # Initialize an empty list to store coverage results
    coverage_results = []

    # Initialize a list to store checkpoints
    checkpoint_path = os.path.join(original_path, "Results", "MAE_results")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, f"{score_type}_{task_name}_checkpoints.pkl")

    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            coverage_results = pickle.load(f)

    # Start the loop from the length of the checkpoint list
    start_index = len(coverage_results)
    # Adjust the loop to start from the start_index
    for i, seed in enumerate(tqdm(seeds[start_index:], desc="Computing coverage for each seed"), start=start_index):
        # checking X_list
        if X_list is not None:
            X = X_list["X"][i]
            theta = X_list["theta"][i]
        else:
            X = None
            theta = None

        coverage_df = compute_coverage(
            score_type=score_type,
            prior_NPE=prior_NPE,
            X = X,
            theta = theta,
            B=B,
            prop_calib=prop_calib,
            alpha=alpha,
            num_obs=num_obs,
            task_name=task_name,
            device=device,
            random_seed=seed,
            min_samples_leaf=min_samples_leaf,
            naive_samples=naive_samples,
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
    X_list = X_dict,
    B=B,
    prop_calib=p_calib,
    alpha=alpha,
    num_obs=500,
    task_name=task_name,
    device=device,
    central_seed=seed,
    min_samples_leaf=300,
    naive_samples=1000,
    n_rep=n_rep,
)

# Create the "MAE_results" folder if it doesn't exist
mae_results_path = os.path.join(original_path, "Results", "MAE_results")
os.makedirs(mae_results_path, exist_ok=True)

# Save the all_coverage_df DataFrame to a CSV file
csv_path = os.path.join(mae_results_path, f"{score_type}_{task_name}_coverage_results.csv")
all_coverage_df.to_csv(csv_path, index=False)

# Compute the summary statistics (mean and standard error) for each column
summary_stats = all_coverage_df.agg(["mean", "std"])
summary_stats.loc["stderr"] = summary_stats.loc["std"] / np.sqrt(n_rep)
summary_stats = summary_stats.drop(index="std")  # Drop standard deviation row

# Save the summary statistics to a CSV file
summary_csv_path = os.path.join(mae_results_path, f"{score_type}_{task_name}_coverage_summary.csv")
summary_stats.to_csv(summary_csv_path)

# Removing all checkpoints
checkpoint_file = os.path.join(mae_results_path, f"{score_type}_{task_name}_checkpoints.pkl")
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
