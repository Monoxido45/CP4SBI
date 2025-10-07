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

import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
from sbi.utils import MultipleIndependent

original_path = os.getcwd()
# Set random seeds for reproducibility
torch.manual_seed(125)
torch.cuda.manual_seed(125)
alpha = 0.1

# defining function to compute cutoffs and compare uncertainty regions between different sim budgets
# B = 5000, 10000 and 20000
def compare_uncertainty_regions(task_name,
                                theta_grid,
                                theta_len,
                                B_list = [5000, 10000, 20000],
                                prop_calib=0.2, 
                                device = "cpu", 
                                min_samples_leaf=[150,300,300], 
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
        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            num_observation=1,
        )[:, :2]

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
        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            num_observation=1,
        )[:, :2]

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
        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            num_observation=1
        )[:, :2]
        
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

        true_post_samples = task._sample_reference_posterior(
            num_samples=1000,
            num_observation=1
        )[:, :2]
    
    uncertainty_map_cdf, uncertainty_map_locart = {}, {}
    cdf_mask, locart_mask = {}, {}
    oracle_mask = {}
    mae_dict_cdf, mae_dict_locart = {}, {}
    target_coverage = 1-alpha

    i = 0
    for B in tqdm(B_list, desc="Making maps for each simulation budget"):
        B_train = int(B * (1 - prop_calib))
        B_calib = int(B * prop_calib)

        theta_train_all = prior(num_samples=B_train)
        X_train = simulator(theta_train_all)
        theta_train_used = theta_train_all[:, :2]

        thetas_calib_all = prior(num_samples=B_calib)
        X_calib = simulator(thetas_calib_all)
        thetas_calib_used = thetas_calib_all[:, :2]

        inference = NPE(prior_NPE, device=device)
        inference.append_simulations(theta_train_used, X_train).train()

        cuda = device == "cuda"

        # fitting locart
        bayes_conf_2d = BayCon(
            sbi_score=HPDScore,
            base_inference=inference,
            is_fitted=True,
            conformal_method="local",
            split_calib=False,
            cuda=device == "cuda",
            alpha=0.1,
        )
        bayes_conf_2d.fit(
            X=X_train,
            theta=theta_train_used,
        )

        res = bayes_conf_2d.locart.sbi_score.compute(X_calib, thetas_calib_used)

        # fitting cdf local
        cdf_conf_2d = BayCon(
            sbi_score=HPDScore,
            base_inference=inference,
            is_fitted=True,
            conformal_method="CDF local",
            cuda=cuda,
            alpha=0.1,
        )
        cdf_conf_2d.fit(
            X=X_train,
            theta=theta_train_used,
        )

        # deriving cutoffs
        bayes_conf_2d.calib(
            X_calib=X_calib,
            theta_calib=res,
            min_samples_leaf=min_samples_leaf[i],
            using_res=True,
        )
        cdf_conf_2d.calib(
            X_calib=X_calib,
            theta_calib=res,
            using_res=True,
            min_samples_leaf=min_samples_leaf[i],
        )

        # obtaining all cutoffs
        locart_cutoff_2d = bayes_conf_2d.predict_cutoff(X_obs)
        cdf_cutoff_2d = cdf_conf_2d.predict_cutoff(X_obs)

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

        mean_coverage_2d_cdf = np.mean(conf_scores_2d <= cdf_cutoff_2d)
        mae_dict_cdf[B] = np.abs(mean_coverage_2d_cdf - target_coverage)

        # computing oracle region for 2d
        t_grid = np.arange(
            np.min(conf_scores_2d),
            np.max(conf_scores_2d),
            0.005,
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

        cdf_unc = cdf_conf_2d.uncertainty_region(
            X=X_obs,
            thetas=theta_grid,
            B=2000,
            beta=0.1,
        )
        locart_unc = locart_unc.reshape(theta_len, theta_len)
        cdf_unc = cdf_unc.reshape(theta_len, theta_len)

        uncertainty_map_locart[B] = locart_unc
        uncertainty_map_cdf[B] = cdf_unc

        log_probs_obs_2d = np.exp(
            post_estim_2d.log_prob(
                x=X_obs.to(device),
                theta=theta_grid.to(device),
            )
            .cpu()
            .numpy()
        )

        # obtaining masks for each method
        cdf_mask_obs = -log_probs_obs_2d < cdf_cutoff_2d
        real_mask_obs = -log_probs_obs_2d < oracle_cutoff_2d
        locart_mask_obs = -log_probs_obs_2d < locart_cutoff_2d

        cdf_mask_obs = cdf_mask_obs.reshape(theta_len, theta_len)
        locart_mask_obs = locart_mask_obs.reshape(theta_len, theta_len)
        real_mask_obs = real_mask_obs.reshape(theta_len, theta_len)

        cdf_mask[B] = cdf_mask_obs
        locart_mask[B] = locart_mask_obs
        oracle_mask[B] = real_mask_obs
        i += 1
    
    # return everything for further plotting
    return [uncertainty_map_cdf, uncertainty_map_locart, cdf_mask, locart_mask, oracle_mask, mae_dict_cdf, mae_dict_locart]

def plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name)   :
    unc_dict_cdf, unc_dict_locart = all_results_list[0], all_results_list[1]
    cdf_mask_dict, locart_mask_dict = all_results_list[2], all_results_list[3]
    real_mask_dict = all_results_list[4]

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, len(unc_dict_cdf), figsize=(5 * len(unc_dict_cdf), 10))
    # axes[0, :] for LOCART, axes[1, :] for CDF
    fig.patch.set_facecolor('black')

    for col_idx, B in enumerate(sorted(unc_dict_cdf.keys())):
        # LOCART row
        ax_locart = axes[0, col_idx]
        locart_unc = unc_dict_locart[B]
        locart_mask_obs = locart_mask_dict[B]
        real_mask_obs = real_mask_dict[B]

        ax_locart.contour(
            locart_mask_obs.T,
            levels=[0.5],
            extent=(-1, 1, -1, 1),
            colors="blue",
            linewidths=2,
            alpha=1.0,
        )
        ax_locart.contourf(
            locart_unc.T,
            levels=[0.99, 1.01],
            extent=(-1, 1, -1, 1),
            colors="lime",
            linewidths=2,
            alpha=0.5,
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

        # CDF row
        ax_cdf = axes[1, col_idx]
        cdf_unc = unc_dict_cdf[B]
        cdf_mask_obs = cdf_mask_dict[B]
        real_mask_obs = real_mask_dict[B]

        ax_cdf.contour(
            cdf_mask_obs.T,
            levels=[0.5],
            extent=(-1, 1, -1, 1),
            colors="dodgerblue",
            linewidths=2,
            alpha=1.0,
        )
        ax_cdf.contourf(
            cdf_unc.T,
            levels=[0.99, 1.01],
            extent=(-1, 1, -1, 1),
            colors="lime",
            linewidths=2,
            alpha=0.5,
        )
        ax_cdf.contourf(
            cdf_unc.T,
            levels=[0.49, 0.51],
            extent=(-1, 1, -1, 1),
            colors="darkorange",
            alpha=0.8,
        )
        ax_cdf.contour(
            real_mask_obs.T,
            levels=[0.5],
            extent=(-1, 1, -1, 1),
            colors="grey",
            linewidths=2,
            alpha=1.0,
        )
        ax_cdf.set_title(f"CDF, B={B}")
        ax_cdf.set_xlabel(r"$\theta_1$")
        ax_cdf.set_ylabel(r"$\theta_2$")
        ax_cdf.set_ylim(y_lims[0], y_lims[1])
        ax_cdf.set_xlim(x_lims[0], x_lims[1])

        # Delete unc and mask objects after using them to free memory
        del locart_unc, locart_mask_obs, cdf_unc, cdf_mask_obs, real_mask_obs
        del ax_locart, ax_cdf
        gc.collect()
        torch.cuda.empty_cache()

    # Add legend only once
    legend_elements = [
        Patch(facecolor="none", edgecolor="dodgerblue", linewidth=2, label=r"$\mathbf{CP4SBI\text{-}CDF}$", alpha=0.75),
        Patch(facecolor="none", edgecolor="blue", linewidth=2, label=r"$\mathbf{CP4SBI\text{-}LOCART}$", alpha=0.75),
        Patch(facecolor="lime", edgecolor="none", linewidth=2, label="Inside region", alpha=0.5),
        Patch(facecolor="darkorange", edgecolor="none", linewidth=2, label="Underterminate region", alpha=0.8),
        Patch(facecolor="none", edgecolor="grey", linewidth=2, label="Oracle region", alpha=1.0),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(legend_elements),
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.rcParams.update({"font.size": 16})
    fig.savefig(f"uncertainty_regions_comparison_{task_name}.pdf", dpi=300)

    plt.close(fig)
    gc.collect()
    # Clean up matplotlib objects to free memory
    del fig, axes, legend_elements
    gc.collect()
    torch.cuda.empty_cache()

# starting by gaussian_linear_uniform
task_name = "gaussian_linear_uniform"
# generating grid of thetas
theta = torch.linspace(-1.005, 1.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(task_name, theta_grid = theta_grid, theta_len = len(theta), seed = 1250)

y_lims = [-0.65, 0.8]
x_lims = [-0.65, 0.85]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "glu")

# testing two moons also
task_name = "two_moons"
# generating grid of thetas
theta = torch.linspace(-1.005, 1.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(task_name, theta_grid = theta_grid, theta_len = len(theta), seed = 1250)

y_lims = [-0.55, 0.05]
x_lims = [-0.1, 0.65]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "two_moons")

# testing for gaussian mixture
task_name = "gaussian_mixture"
# generating grid of thetas
theta = torch.linspace(-3.005, 3.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta), 
    seed = 1250,)

y_lims = [-1.05, 1.05]
x_lims = [-1.15, 1.15]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "gaussian_mixture")

# testing another seed
task_name = "gaussian_mixture"
# generating grid of thetas
theta = torch.linspace(-3.005, 3.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

all_results_list = compare_uncertainty_regions(
    task_name, 
    theta_grid = theta_grid, 
    theta_len = len(theta), 
    B_list = [5000, 10000, 20000, 30000],
    seed = 750,)

y_lims = [-1.15, 0.75]
x_lims = [-1.25, 1.25]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "gaussian_mixture_v2")

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

y_lims = [-0.3, 0.25]
x_lims = [-0.1, 0.65]
plot_uncertainty_regions(all_results_list, x_lims, y_lims, task_name = "gaussian_linear")




