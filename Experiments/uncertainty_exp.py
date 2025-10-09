import numpy as np
import os
import torch

from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from sbi.utils.user_input_checks import process_prior
import sbibm
from copy import deepcopy
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pickle
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
from sbi.utils import MultipleIndependent

original_path = os.getcwd()
# Set random seeds for reproducibility
torch.manual_seed(125)
torch.cuda.manual_seed(125)
alpha = 0.1


task = sbibm.get_task("sir")
simulator = task.get_simulator()
prior = task.get_prior()

# defining function to compute cutoffs and compare uncertainty regions between different calibration budgets
# B = 1000, 2000, 4000, 6000
# fixing B = 10000 for NPE training
def compare_uncertainty_regions(task_name,
                                theta_grid,
                                theta_len,
                                B_list = [1000, 2000, 4000, 6000],
                                B_train = 10000,
                                device = "cpu", 
                                min_samples_leaf=[150,300,300,300],
                                X_str = False,
                                seed = 125,):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if task_name != "gaussian_mixture":
        task = sbibm.get_task(task_name)
        simulator = task.get_simulator()
        prior = task.get_prior()
    else:
        print("Using custom Gaussian Mixture task")
        from CP4SBI.gmm_task import GaussianMixture

        task = GaussianMixture(dim=2, prior_bound=3.0)
        simulator = task.get_simulator()
        prior = task.get_prior()
    
    # Load the dictionary from the pickle file
    if X_str:
        posterior_data_path = (
            original_path + f"/Results/posterior_data/{task_name}_posterior_samples.pkl"
        )
        with open(posterior_data_path, "rb") as f:
            X_dict = pickle.load(f)
    
        # Load the X_list pickle file from the X_data folder
        if B_train >= 20000:
            x_data_path = os.path.join(
            original_path, "Results/X_data", f"{task_name}_X_samples_30000.pkl"
        )
            with open(x_data_path, "rb") as f:
                X_data = pickle.load(f)

                 # Load the X_list pickle file from the X_data folder
            theta_data_path = os.path.join(
                original_path, "Results/X_data", f"{task_name}_theta_samples_30000.pkl"
            )
            with open(theta_data_path, "rb") as f:
                theta_list = pickle.load(f)
        else:
            x_data_path = os.path.join(
                original_path, "Results/X_data", f"{task_name}_X_samples_20000.pkl"
            )
            with open(x_data_path, "rb") as f:
                X_data = pickle.load(f)

            # Load the X_list pickle file from the X_data folder
            theta_data_path = os.path.join(
                original_path, "Results/X_data", f"{task_name}_theta_samples_20000.pkl"
            )
            with open(theta_data_path, "rb") as f:
                theta_list = pickle.load(f)

        X_list = {"X": X_data, "theta": theta_list}

        X = X_list["X"][0]
        theta = X_list["theta"][0]
        # splitting X
        indices = torch.randperm(X.shape[0])
        train_indices = indices[:B_train]
        calib_indices = indices[B_train:]

        X_train = X[train_indices]
        X_calib = X[calib_indices]

        # splitting theta
        theta_train_all = theta[train_indices]
        thetas_calib_all = theta[calib_indices]
    
    else:
        theta_all = prior(num_samples= 30000)
        X_all = simulator(theta_all)

        # splitting X
        indices = torch.randperm(X_all.shape[0])
        train_indices = indices[:B_train]
        calib_indices = indices[B_train:]
        X_train = X_all[train_indices]
        X_calib = X_all[calib_indices]
        # splitting theta
        theta_train_all = theta_all[train_indices]
        thetas_calib_all = theta_all[calib_indices]

    # determining prior for NPE
    if task_name == "two_moons":
        prior_NPE = BoxUniform(
            low=-1 * torch.ones(2),
            high=1 * torch.ones(2),
            device=device,
        )
        theta_real = torch.full((1, 2), 0.0)
        theta_real[0, 0] = 0.1
        theta_real[0, 1] = -0.3
        X_obs = simulator(theta_real)
        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            num_observation=20,
            observation=X_obs,
        )

    elif task_name == "gaussian_linear_uniform":
        prior_NPE = BoxUniform(
            low=-1 * torch.ones(2),
            high=1 * torch.ones(2),
            device=device,
        )
        theta_real = torch.full((1, 10), 0.0)
        theta_real[0, 0] = 0.25
        theta_real[0, 1] = 0.1
        X_obs = simulator(theta_real)
        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            observation=X_obs,
        )[:,:2]

    elif task_name == "slcp" or task_name == "slcp_distractors":
        prior_NPE = BoxUniform(
            low=-3 * torch.ones(2),
            high=3 * torch.ones(2),
            device=device,
        )
        X_obs = task.get_observation(num_observation=1)
        first_entry = next(iter(X_dict))
        true_post_samples = X_dict[first_entry][:, :2]

    elif task_name == "gaussian_linear":
        prior_params = {
            "loc": torch.zeros((2,), device=device),
            "precision_matrix": torch.inverse(
                0.1 * torch.eye(2, device=device)
            ),
        }
        prior_dist = MultivariateNormal(
            **prior_params,
            validate_args=False,
        )
        prior_NPE, _, _ = process_prior(prior_dist)

        theta_real = torch.full((1, 10), 0.0)
        theta_real[0, 0] = 0.25
        theta_real[0, 1] = 0.1 
        X_obs = simulator(theta_real)
        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            observation=X_obs,
        )[:, :2]

    elif task_name == "bernoulli_glm" or task_name == "bernoulli_glm_raw":
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
        
        # taking one of the observations with ground truth available
        X_obs = task.get_observation(num_observation=1)
        first_entry = next(iter(X_dict))
        true_post_samples = X_dict[first_entry][:, :2]

    elif task_name == "gaussian_mixture":
        prior_NPE = BoxUniform(
            low=-3 * torch.ones(2),
            high=3 * torch.ones(2),
            device=device,
        )
        theta_real = torch.full((1, 2), 0.0)
        theta_real[0, 0] = 0.15
        theta_real[0, 1] = -0.1
        X_obs = simulator(theta_real)
        true_post_samples = task._sample_reference_posterior(
        num_samples=1000,
        observation=X_obs,
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
        X_obs = task.get_observation(num_observation=1)
        first_entry = next(iter(X_dict))
        true_post_samples = X_dict[first_entry][:, :2]
        print(true_post_samples)
      
    elif task_name == "lotka_volterra":
        mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        prior_params = {
            "loc": torch.tensor([mu_p1, mu_p2], device=device),
            "scale": torch.tensor([sigma_p, sigma_p], device=device),
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
            )
        ]
        prior_dist = MultipleIndependent(prior_list, validate_args=False)
        prior_NPE, _, _ = process_prior(prior_dist)
        X_obs = task.get_observation(num_observation=1)
        first_entry = next(iter(X_dict))
        true_post_samples = X_dict[first_entry][:, :2]

    uncertainty_map_locart, locart_mask = {}, {}
    oracle_mask, mae_dict_locart = {}, {}
    target_coverage = 1-alpha
    
    error_locart_dict = {}

    theta_train_used = theta_train_all[:, :2]
    # training the NPE only once with B = 10000
    inference = NPE(prior_NPE, device=device)
    inference.append_simulations(theta_train_used, X_train).train()

    i = 0
    for B in tqdm(B_list, desc="Making maps for each simulation budget"):
        X_calib_used = X_calib[:B, :]
        thetas_calib_used = thetas_calib_all[:B, :2]

        # fitting locart
        bayes_conf_2d = BayCon(
            sbi_score=HPDScore,
            base_inference=inference,
            is_fitted=True,
            conformal_method="local",
            split_calib=False,
            weighting=True,
            cuda=device == "cuda",
            alpha=0.1,
        )
        bayes_conf_2d.fit(
            X=X_train,
            theta=theta_train_used,
        )

        res = bayes_conf_2d.locart.sbi_score.compute(X_calib_used, thetas_calib_used)

        # deriving cutoffs
        bayes_conf_2d.calib(
            X_calib=X_calib_used,
            theta_calib=res,
            min_samples_leaf=min_samples_leaf[i],
            using_res=True,
        )

        # obtaining all cutoffs
        locart_cutoff_2d = bayes_conf_2d.predict_cutoff(X_obs)

        post_estim_2d = deepcopy(bayes_conf_2d.locart.sbi_score.posterior)

        # coverage for 2d
        post_samples_2d = true_post_samples[:, :2]
        conf_scores_2d = -np.exp(
            post_estim_2d.log_prob(
                post_samples_2d.to(device=device),
                x=X_obs.to(device=device),
            )
            .cpu()
            .numpy()
        )

        mean_coverage_2d = np.mean(conf_scores_2d <= locart_cutoff_2d)
        mae_dict_locart[B] = np.abs(mean_coverage_2d - target_coverage)

        # computing oracle region for 2d
        t_grid = np.arange(
            np.min(conf_scores_2d),
            np.max(conf_scores_2d),
            0.01,
        )

        # computing MC integral for all t_grid
        coverage_array = np.zeros(t_grid.shape[0])
        for t in t_grid:
            coverage_array[t_grid == t] = np.mean(conf_scores_2d <= t)

        closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
        # finally, finding the naive cutoff
        oracle_cutoff_2d = t_grid[closest_t_index]        

        locart_unc = bayes_conf_2d.uncertainty_region(
            X=X_obs,
            thetas=theta_grid,
            beta=0.1,
        )

        print(bayes_conf_2d.cutoff_CI)
        print(locart_cutoff_2d)
        locart_unc = locart_unc.reshape(theta_len, theta_len)

        uncertainty_map_locart[B] = locart_unc

        log_probs_obs_2d = np.exp(
            post_estim_2d.log_prob(
                x=X_obs.to(device),
                theta=theta_grid.to(device),
            )
            .cpu()
            .numpy()
        )

        # obtaining masks for each method
        real_mask_obs = -log_probs_obs_2d < oracle_cutoff_2d
        locart_mask_obs = -log_probs_obs_2d < locart_cutoff_2d

        locart_mask_obs = locart_mask_obs.reshape(theta_len, theta_len)
        real_mask_obs = real_mask_obs.reshape(theta_len, theta_len)

        # computing inside and outside regions for conf_scores_2d thetas
        locart_unc_true = bayes_conf_2d.uncertainty_region(
            X=X_obs,
            thetas=post_samples_2d.to(device=device),
            beta=0.1,
        )


        # obtaining prob of each kind of error by using true post samples
        not_in_oracle_region = conf_scores_2d > oracle_cutoff_2d
        # computing type 2 error for locart and CDF
        not_in_oracle_but_in_locart = np.logical_and(
            not_in_oracle_region, locart_unc_true > 0.99
        )
        type_2_error_locart = np.sum(not_in_oracle_but_in_locart)/np.sum(not_in_oracle_region)


        in_oracle_region = conf_scores_2d <= oracle_cutoff_2d

        # computing type 1 error for locart and CDF
        in_oracle_but_not_in_locart = np.logical_and(
            in_oracle_region, locart_unc_true < 0.01
        )
        type_1_error_locart = np.sum(in_oracle_but_not_in_locart) / np.sum(in_oracle_region)

        
        print(f"B={B}: Type 1 error LOCART={type_1_error_locart:.4f}, Type 2 error LOCART={type_2_error_locart:.4f}")
    
        error_locart_dict[B] = [type_1_error_locart, type_2_error_locart]

        locart_mask[B] = locart_mask_obs
        oracle_mask[B] = real_mask_obs
        i += 1
    
    # return everything for further plotting
    return [uncertainty_map_locart, locart_mask, oracle_mask, mae_dict_locart, error_locart_dict]

def plot_uncertainty_regions(
        all_results_list, 
        x_lims, 
        y_lims, 
        task_name)   :
    # Only LOCART results are available, so update unpacking and plotting accordingly
    unc_dict_locart = all_results_list[0]
    locart_mask_dict = all_results_list[1]
    real_mask_dict = all_results_list[2]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, len(unc_dict_locart), figsize=(5 * len(unc_dict_locart), 5))
    fig.patch.set_facecolor('black')

    # If only one budget, axes is not an array
    if len(unc_dict_locart) == 1:
        axes = [axes]

    for col_idx, B in enumerate(sorted(unc_dict_locart.keys())):
        ax_locart = axes[col_idx]
        locart_unc = unc_dict_locart[B]
        locart_mask_obs = locart_mask_dict[B]
        real_mask_obs = real_mask_dict[B]

        ax_locart.contour(
            locart_mask_obs.T,
            levels=[0.5],
            extent=(-1, 1, -1, 1),
            colors="dodgerblue",
            linewidths=2,
            alpha=1.0,
        )
        ax_locart.contourf(
            locart_unc.T,
            levels=[0.99, 1.01],
            extent=(-1, 1, -1, 1),
            colors="lime",
            linewidths=2,
            alpha=0.25,
        )
        ax_locart.contourf(
            locart_unc.T,
            levels=[0.49, 0.51],
            extent=(-1, 1, -1, 1),
            colors="darkorange",
            alpha=0.8,
        )
        ax_locart.contour(
            real_mask_obs.T,
            levels=[0.5],
            extent=(-1, 1, -1, 1),
            colors="grey",
            linewidths=2,
            alpha=1.0,
        )
        ax_locart.set_title(f"LOCART, B={B}")
        ax_locart.set_xlabel(r"$\theta_1$")
        ax_locart.set_ylabel(r"$\theta_2$")
        ax_locart.set_ylim(y_lims[0], y_lims[1])
        ax_locart.set_xlim(x_lims[0], x_lims[1])

        del locart_unc, locart_mask_obs, real_mask_obs, ax_locart
        gc.collect()
        torch.cuda.empty_cache()

    legend_elements = [
        Patch(facecolor="none", edgecolor="dodgerblue", linewidth=2, label=r"$\mathbf{CP4SBI\text{-}LOCART}$", alpha=0.75),
        Patch(facecolor="lime", edgecolor="none", linewidth=2, label="Inside region", alpha=0.25),
        Patch(facecolor="darkorange", edgecolor="none", linewidth=2, label="Underterminate region", alpha=0.8),
        Patch(facecolor="none", edgecolor="grey", linewidth=2, label="Target region", alpha=1.0),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(legend_elements),
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.825)
    plt.rcParams.update({"font.size": 16})
    fig.savefig(f"uncertainty_regions_comparison_{task_name}.pdf", dpi=300)

    plt.close(fig)
    gc.collect()
    del fig, axes, legend_elements
    gc.collect()
    torch.cuda.empty_cache()

# starting by gaussian_linear_uniform
task_name = "gaussian_linear_uniform"
# generating grid of thetas
theta = torch.linspace(-1.005, 1.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    B_train = 5000,
    seed = 750,
)
with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [-0.15, 1.25]
x_lims = [-0.55,1.0]
plot_uncertainty_regions(
    all_results_list, 
    x_lims, 
    y_lims, 
    task_name = "gaussian_linear_uniform",
    )

# testing two moons also
task_name = "two_moons"
# generating grid of thetas
theta = torch.linspace(-1.005, 1.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    B_train = 5000,
    seed = 750,
    )
with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [-0.55, 0.]
x_lims = [-0.15, 0.65]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "two_moons")

# testing for gaussian mixture
task_name = "gaussian_mixture"
# generating grid of thetas
theta = torch.linspace(-3.005, 3.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name,
    theta_grid=theta_grid,
    theta_len=len(theta),
    B_train=10000,
    seed=750,
)

with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [-1.05, 1.05]
x_lims = [-1.2, 1.15]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "gaussian_mixture")

# testing for gaussian linear
task_name = "gaussian_linear"
# generating grid of thetas
theta = torch.linspace(-2.005, 2.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    seed = 1250,)

with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [-0.3, 0.3]
x_lims = [-0.15, 0.55]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "gaussian_linear")

task_name = "slcp"
# generating grid of thetas
theta = torch.linspace(-3.005, 3.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    X_str=True,
    B_train = 20000,
    seed = 1250,)

with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [-1.15, 1.15]
x_lims = [-1.15, 1.15]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "slcp")

task_name = "bernoulli_glm"
# generating grid of thetas
theta = torch.linspace(-3.000, 3.000, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    X_str=True,
    seed = 750,)

with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [0.2, 1.05]
x_lims = [-0.1, 0.65]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "bernoulli_glm")

task_name = "sir"
# generating grid of thetas
theta = torch.linspace(-0.05, 2.05, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    B_train = 20000,
    X_str=True,
    seed = 750,)

with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [0.12, 0.25]
x_lims = [0.6, 0.7]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "sir")

task_name = "lotka_volterra"
# generating grid of thetas
theta = torch.linspace(0.005, 2.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta),
    B_train = 20000,
    X_str=True,
    seed = 750,)

with open(f"all_results_list_{task_name}.pkl", "wb") as f:
    pickle.dump(all_results_list, f)

y_lims = [0, 1.15]
x_lims = [0, 1.15]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "lotka_volterra")

