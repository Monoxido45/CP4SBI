# for posterior estimation and calibration
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE, NPSE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore, WALDOScore, KDE_HPDScore
from sbi.utils.user_input_checks import process_prior
from sbi.utils import MultipleIndependent
from CP4SBI.utils import naive_method, hdr_method
from scipy.stats import gaussian_kde

# for benchmarking
import sbibm
from copy import deepcopy
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal

# for plotting and broadcasting
from tqdm import tqdm
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
    type=float,
)

parser.add_argument(
    "--sample_with",
    "-sw",
    help="string for sampling method to be used inside direct_posterior",
    default="direct",
    type=str,
)

parser.add_argument(
    "--n_x",
    "-nx",
    help="number of X samples to be used",
    default=500,
    type=int,
)

parser.add_argument(
    "--base_model",
    "-b_m",
    help="string for base model to be used",
    default="NPE",
    type=str,
)

parser.add_argument(
    "--min_samples_leaf",
    "-m_s_l",
    help="int for minimum number of samples in leaf for local methods",
    default=300,
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
X_str = args.X_list == "True"
num_obs = args.n_x
sample_with = args.sample_with
base_model = args.base_model
min_samples_leaf = args.min_samples_leaf

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

# Set the random seed for reproducibility
alpha = 0.1

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


# Defining prior for NPE and volume grid for each case
# only considering 2d cases for sake of easier volume calculation
if task_name == "two_moons":
    prior_NPE = BoxUniform(
        low=-1 * torch.ones(2),
        high=1 * torch.ones(2),
        device=device,
    )

    # deifining area of the bounding box for volume calculation
    area_box = 4

    # sample uniformly from the BoxUniform prior instead of a regular linspace grid
    n_samples = 5000

    # use prior sampling if available (respects the prior)
    eval_grid = prior_NPE.sample((n_samples,))

elif task_name == "gaussian_mixture":
    prior_NPE = BoxUniform(
        low=-3 * torch.ones(2),
        high=3 * torch.ones(2),
        device=device,
    )

    # defining area of the bounding box for volume calculation
    area_box = 36
    
    # sample uniformly from the BoxUniform prior instead of a regular linspace grid
    n_samples = 5000

    # use prior sampling if available (respects the prior)
    eval_grid = prior_NPE.sample((n_samples,))

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

    # defining a bounding box between 0 and 5 for theta_1 and between 0 and 1 for theta_2
    area_box = 5
    dist_box = BoxUniform(
        low= torch.tensor([0.0, 0.0]),
        high= torch.tensor([5.0, 1.0]),
        device=device,
    )

    n_samples = 5000

    # use prior sampling if available (respects the prior)
    eval_grid = dist_box.sample((n_samples,))


def compute_volume(
    prior_NPE,
    score_type,
    eval_grid,
    split_calib=False,
    X=None,
    theta=None,
    B=5000,
    prop_calib=0.2,
    alpha=0.1,
    num_obs=500,
    task_name="two_moons",
    device="cuda",
    random_seed=0,
    min_samples_leaf=300,
    naive_samples=1000,
    sample_with="direct",
):
    # setting seed
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

    if base_model == "NPE":
        # fitting NPE
        inference = NPE(prior_NPE, device=device, summary_writer=None)
        inference.append_simulations(
            theta_train,
            X_train,
        ).train()

    elif base_model == "NPSE":
        # fitting diffusion model
        inference = NPSE(prior_NPE, device=device, summary_writer=None)
        inference.append_simulations(
            theta=theta_train.to(device),
            x=X_train.to(device),
        ).train()

    cuda = device == "cuda"

    # checking score type
    if score_type == "HPD":
        score_used = HPDScore
        kde_use = False
    elif score_type == "WALDO":
        score_used = WALDOScore
        kde_use = False
    elif score_type == "KDE":
        score_used = KDE_HPDScore
        kde_use = True

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

    volume_locart = np.zeros(num_obs)
    volume_global = np.zeros(num_obs)
    volume_cdf = np.zeros(num_obs)
    volume_naive = np.zeros(num_obs)
    volume_local_cdf = np.zeros(num_obs)
    volume_a_locart = np.zeros(num_obs)

    post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)
    volume_hdr = np.zeros(num_obs)

    # Load the dictionary from the pickle file
    posterior_data_path = (
        original_path + f"/Results/posterior_data/{task_name}_posterior_samples.pkl"
    )
    with open(posterior_data_path, "rb") as f:
        X_dict = pickle.load(f)

    X_obs = torch.cat(list(X_dict.keys())).numpy()
 
    # HDR recalibration
    print("Fitting HDR recalibration")
    if score_type == "HPD" or score_type == "KDE":
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
            kde=kde_use,
        )

    locart_cutoff = bayes_conf.predict_cutoff(X_obs)
    global_cutoff = global_conf.predict_cutoff(X_obs)
    cdf_cutoff = cdf_conf.predict_cutoff(X_obs)
    local_cdf_cutoff = local_cdf_conf.predict_cutoff(X_obs)
    alocart_cutoff = w_bayes_conf.predict_cutoff(X_obs)

    i = 0
    dict_keys = list(X_dict.keys())
    # evaluating cutoff for each observation
    for X_0 in tqdm(dict_keys, desc="Computing volume across observations"):
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
            if len(X_0.shape) == 1:
                X_0 = X_0.reshape(1, -1)

            # computing scores for naive, locart, global, cdf, local_cdf, and alocart
            conf_scores = -np.exp(
                post_estim.log_prob(
                    eval_grid.to(device=device),
                    x=X_0.to(device=device),
                )
                .cpu()
                .numpy()
            )

            # computing scores for HDR
            _, dens_samples = hdr_obj.recal_sample(
                y_hat=eval_grid.reshape(
                    1,
                    eval_grid.shape[0],
                    eval_grid.shape[1],
                ),
                f_hat_y_hat=-conf_scores.reshape(1, -1),
            )

            hdr_conf_scores = -dens_samples[0, :]

        elif score_type == "KDE":
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
                    score_type="HPD",
                    device=device,
                    n_grid=1000,
                    B_naive=naive_samples,
                    kde=True,
                )
            else:
                closest_t = naive_method(
                    post_estim,
                    X=X_0,
                    alpha=alpha,
                    score_type="HPD",
                    device=device,
                    B_naive=naive_samples,
                    kde=True,
                )
            if len(X_0.shape) == 1:
                X_0 = X_0.reshape(1, -1)

            sample_generated = (
                post_estim.sample(
                    (1000,),
                    x=X_0,
                    show_progress_bars=False,
                )
                .cpu()
                .detach()
                .numpy()
            )

            #  fitting KDE
            kde = gaussian_kde(sample_generated.T, bw_method="scott")

            # computing log_prob for only one X
            conf_scores = -kde(eval_grid.T)

            # computing scores for HDR
            _, dens_samples = hdr_obj.recal_sample(
                y_hat=eval_grid.reshape(
                    1,
                    eval_grid.shape[0],
                    eval_grid.shape[1],
                ),
                f_hat_y_hat=-conf_scores.reshape(1, -1),
            )

            hdr_conf_scores = -dens_samples[0, :]
        
        # computing volume
        volume_locart[i] = np.sum(conf_scores <= locart_cutoff[i])/conf_scores.shape[0] * area_box
        volume_global[i] = np.sum(conf_scores <= global_cutoff[i])/conf_scores.shape[0] * area_box
        volume_cdf[i] = np.sum(conf_scores <= cdf_cutoff[i])/conf_scores.shape[0] * area_box
        volume_local_cdf[i] = np.sum(conf_scores <= local_cdf_cutoff[i])/conf_scores.shape[0] * area_box
        volume_a_locart[i] = np.sum(conf_scores <= alocart_cutoff[i])/conf_scores.shape[0] * area_box
        volume_hdr[i] = np.sum(hdr_conf_scores <= hdr_cutoff[i])/hdr_conf_scores.shape[0] * area_box
        volume_naive[i] = np.sum(conf_scores <= closest_t)/conf_scores.shape[0] * area_box

        i += 1

    # Creating a pandas DataFrame with the mean volume values
    volume_df = pd.DataFrame(
        {
            "LOCART vol": [np.mean((volume_locart))],
            "A-LOCART vol": [np.mean((volume_a_locart))],
            "Global CP vol": [np.mean((volume_global))],
            "Naive vol": [np.mean((volume_naive))],
            "CDF vol": [np.mean((volume_cdf))],
            "Local CDF vol": [np.mean((volume_local_cdf))],
            "HDR vol": [np.mean((volume_hdr))],
        }
    )
    return volume_df

def compute_volume_repeated(
    prior_NPE,
    score_type,
    eval_grid,
    X_list=None,
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
    sample_with="direct",
):
    # Generate an array of seeds using the central_seed
    seeds = np.random.RandomState(central_seed).randint(0, 2**32 - 1, size=n_rep)

    # Initialize an empty list to store volume results
    volume_results = []

    # Initialize a list to store checkpoints
    if base_model == "NPE":
        checkpoint_path = os.path.join(original_path, "Results", "volume_results")
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_path, f"{score_type}_{task_name}_checkpoints.pkl"
        )
    else:
        checkpoint_path = os.path.join(
            original_path, "Results", f"volume_results_{base_model}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            checkpoint_path, f"{score_type}_{task_name}_checkpoints.pkl"
        )
        print(checkpoint_file)

    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            volume_results = pickle.load(f)

    # Start the loop from the length of the checkpoint list
    start_index = len(volume_results)
    if task_name == "gaussian_mixture":
        torch.manual_seed(central_seed)
        torch.cuda.manual_seed(central_seed)

    # Adjust the loop to start from the start_index
    for j, seed in enumerate(
        tqdm(seeds[start_index:], desc="Computing volume for each seed"),
        start=start_index,
    ):
        # checking X_list
        if X_list is not None:
            X = X_list["X"][j]
            theta = X_list["theta"][j]
        else:
            X = None
            theta = None

        volume_df = compute_volume(
            score_type=score_type,
            prior_NPE=prior_NPE,
            eval_grid=eval_grid,
            X=X,
            theta=theta,
            B=B,
            prop_calib=prop_calib,
            alpha=alpha,
            num_obs=num_obs,
            task_name=task_name,
            device=device,
            random_seed=seed,
            min_samples_leaf=min_samples_leaf,
            naive_samples=naive_samples,
            sample_with=sample_with,
        )
        volume_results.append(volume_df)

        with open(checkpoint_file, "wb") as f:
            pickle.dump(volume_results, f)

    # Combine results into a single DataFrame
    combined_volume_df = pd.concat(volume_results, ignore_index=True)
    return combined_volume_df



all_volume_df = compute_volume_repeated(
    score_type=score_type,
    prior_NPE=prior_NPE,
    eval_grid=eval_grid,
    X_list=X_list,
    B=B,
    prop_calib=p_calib,
    alpha=alpha,
    num_obs=num_obs,
    task_name=task_name,
    device=device,
    central_seed=seed,
    min_samples_leaf=min_samples_leaf,
    naive_samples=1000,
    n_rep=n_rep,
    sample_with=sample_with,
)

# Create the "volume_results" folder if it doesn't exist for NPE
if base_model == "NPE":
    volume_results_path = os.path.join(original_path, "Results", "volume_results")
    os.makedirs(volume_results_path, exist_ok=True)
else:
    volume_results_path = os.path.join(
        original_path, "Results", f"volume_results_{base_model}"
    )
    os.makedirs(volume_results_path, exist_ok=True)

# Save the all_volume_df DataFrame to a CSV file
csv_path = os.path.join(
    volume_results_path, f"{score_type}_{task_name}_volume_results_{B}.csv"
)
all_volume_df.to_csv(csv_path, index=False)

# Compute the summary statistics (mean and standard error) for each column
summary_stats = all_volume_df.agg(["mean", "std"])
summary_stats.loc["stderr"] = summary_stats.loc["std"] / np.sqrt(n_rep)
summary_stats = summary_stats.drop(index="std")  # Drop standard deviation row

# Save the summary statistics to a CSV file
summary_csv_path = os.path.join(
    volume_results_path, f"{score_type}_{task_name}_volume_summary_{B}.csv"
)
summary_stats.to_csv(summary_csv_path)

# Removing all checkpoints
checkpoint_file = os.path.join(
    volume_results_path, f"{score_type}_{task_name}_checkpoints.pkl"
)
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
