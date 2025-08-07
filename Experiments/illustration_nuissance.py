import numpy as np
import pickle
import os
import torch

from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from CP4SBI.utils import naive_method
from tqdm import tqdm
import sbibm
from copy import deepcopy
from matplotlib.patches import Patch
from matplotlib.legend import Legend

import matplotlib.pyplot as plt

original_path = os.getcwd()
# Set random seeds for reproducibility
torch.manual_seed(125)
torch.cuda.manual_seed(125)
alpha = 0.1

task_name = "gaussian_linear_uniform"
device = "cpu"

B = 20000
prop_calib = 0.2
B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)

prior_NPE_2d = BoxUniform(
    low=-1 * torch.ones(2),
    high=1 * torch.ones(2),
    device=device,
)

prior_NPE_1d = BoxUniform(
    low=-1 * torch.ones(1),
    high=1 * torch.ones(1),
    device=device,
)

task = sbibm.get_task(task_name)
simulator = task.get_simulator()
prior = task.get_prior()

B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)

# Generate training and calibration data
# first for 2d
theta_train_all = prior(num_samples=B_train)
X_train = simulator(theta_train_all)
theta_train_2d = theta_train_all[:, :2]

thetas_calib_all = prior(num_samples=B_calib)
X_calib = simulator(thetas_calib_all)
thetas_calib_2d = thetas_calib_all[:, :2]

# then for 1d
theta_train_1d = theta_train_all[:, :1]
thetas_calib_1d = thetas_calib_all[:, :1]

# Fit NPE for 1d and 2d separately
# Fitting NPE for 2d
inference_2d = NPE(prior_NPE_2d, device=device)
inference_2d.append_simulations(theta_train_2d, X_train).train()

# Fit all calibration methods for each dimension
cuda = device == "cuda"
# LOCART for 2d
bayes_conf_2d = BayCon(
    sbi_score=HPDScore,
    base_inference=inference_2d,
    is_fitted=True,
    conformal_method="local",
    split_calib=False,
    cuda=device == "cuda",
    alpha=0.1,
)
bayes_conf_2d.fit(
    X=X_train,
    theta=theta_train_2d,
)

cdf_conf_2d = BayCon(
    sbi_score=HPDScore,
    base_inference=inference_2d,
    is_fitted=True,
    conformal_method="CDF",
    cuda=cuda,
    alpha=0.1,
)
cdf_conf_2d.fit(
    X=X_train,
    theta=theta_train_1d,
)

# Compute conformity scores for calibration data
res = bayes_conf_2d.locart.sbi_score.compute(X_calib, thetas_calib_2d)

# Calibrate LOCART
bayes_conf_2d.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)

cdf_conf_2d.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)

global_conf_2d = BayCon(
    sbi_score=HPDScore,
    base_inference=inference_2d,
    is_fitted=True,
    conformal_method="global",
    cuda=cuda,
    alpha=alpha,
)

global_conf_2d.fit(
    X=X_train,
    theta=theta_train_2d,
)

global_conf_2d.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)

# Fitting NPE for 1d
inference_1d = NPE(prior_NPE_1d, device=device)
inference_1d.append_simulations(theta_train_1d, X_train).train()
# LOCART for 1d
bayes_conf_1d = BayCon(
    sbi_score=HPDScore,
    base_inference=inference_1d,
    is_fitted=True,
    conformal_method="local",
    split_calib=False,
    cuda=cuda,
    alpha=0.1,
)
bayes_conf_1d.fit(
    X=X_train,
    theta=theta_train_1d,
)


cdf_conf_1d = BayCon(
    sbi_score=HPDScore,
    base_inference=inference_1d,
    is_fitted=True,
    conformal_method="CDF",
    cuda=cuda,
    alpha=0.1,
)
cdf_conf_1d.fit(
    X=X_train,
    theta=theta_train_1d,
)

# Compute conformity scores for calibration data
res = bayes_conf_1d.locart.sbi_score.compute(X_calib, thetas_calib_1d)

# Calibrate LOCART
bayes_conf_1d.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)

cdf_conf_1d.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)

global_conf_1d = BayCon(
    sbi_score=HPDScore,
    base_inference=inference_1d,
    is_fitted=True,
    conformal_method="global",
    cuda=cuda,
    alpha=alpha,
)

global_conf_1d.fit(
    X=X_train,
    theta=theta_train_1d,
)

global_conf_1d.calib(
    X_calib=X_calib,
    theta_calib=res,
    using_res=True,
)

############################### Obtaining coverage deviation
post_estim_1d = deepcopy(bayes_conf_1d.locart.sbi_score.posterior)
post_estim_2d = deepcopy(bayes_conf_2d.locart.sbi_score.posterior)

target_coverage = 0.9
torch.manual_seed(125)
torch.cuda.manual_seed(125)
# first X_obs
theta_real = torch.full((1, 10), 0.0)
theta_real[0, 0] = 0.25
theta_real[0, 1] = 0.1

theta_2d = theta_real[:, :2]
theta_1d = theta_real[:, :1]

# generating X_obs
X_obs = simulator(theta_real)

# obtaining all cutoffs
locart_cutoff_2d = bayes_conf_2d.predict_cutoff(X_obs)
global_cutoff_2d = global_conf_2d.predict_cutoff(X_obs)
cdf_cutoff_2d = cdf_conf_2d.predict_cutoff(X_obs)
locart_cutoff_1d = bayes_conf_1d.predict_cutoff(X_obs)
global_cutoff_1d = global_conf_1d.predict_cutoff(X_obs)
cdf_cutoff_1d = cdf_conf_1d.predict_cutoff(X_obs)

# obtaining naive cutoffs
naive_cutoff_2d = naive_method(
    post_estim_2d,
    X=X_obs,
    alpha=0.1,
    score_type="HPD",
    device=device,
    B_naive=1000,
)

naive_cutoff_1d = naive_method(
    post_estim_1d,
    X=X_obs,
    alpha=0.1,
    score_type="HPD",
    device=device,
    B_naive=1000,
)

# computing coverage difference for 2d and 1d
true_post_samples = task._sample_reference_posterior(
    num_samples=1000,
    observation=X_obs,
)

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
mae_2d_locart = np.abs(mean_coverage_2d - target_coverage)

mean_coverage_2d_global = np.mean(conf_scores_2d <= global_cutoff_2d)
mae_2d_global = np.abs(mean_coverage_2d_global - target_coverage)

mean_coverage_2d_naive = np.mean(conf_scores_2d <= naive_cutoff_2d)
mae_2d_naive = np.abs(mean_coverage_2d_naive - target_coverage)

mean_coverage_2d_cdf = np.mean(conf_scores_2d <= cdf_cutoff_2d)
mae_2d_cdf = np.abs(mean_coverage_2d_cdf - target_coverage)

print("MAE 2D CDF:", mae_2d_cdf)
print("MAE 2D LOCART:", mae_2d_locart)
print("MAE 2D Global:", mae_2d_global)
print("MAE 2D Naive:", mae_2d_naive)

# coverage for 1d
post_samples_1d = true_post_samples[:, :1]
conf_scores_1d = -np.exp(
    post_estim_1d.log_prob(
        post_samples_1d.to(device=device),
        x=X_obs.to(device=device),
    )
    .cpu()
    .numpy()
)

mean_coverage_1d = np.mean(conf_scores_1d <= locart_cutoff_1d)
mae_1d_locart = np.abs(mean_coverage_1d - target_coverage)

mean_coverage_1d_global = np.mean(conf_scores_1d <= global_cutoff_1d)
mae_1d_global = np.abs(mean_coverage_1d_global - target_coverage)

mean_coverage_1d_naive = np.mean(conf_scores_1d <= naive_cutoff_1d)
mae_1d_naive = np.abs(mean_coverage_1d_naive - target_coverage)

mean_coverage_1d_cdf = np.mean(conf_scores_1d <= cdf_cutoff_1d)
mae_1d_cdf = np.abs(mean_coverage_1d_cdf - target_coverage)

print("MAE 1D CDF:", mae_1d_cdf)
print("MAE 1D LOCART:", mae_1d_locart)
print("MAE 1D Global:", mae_1d_global)
print("MAE 1D Naive:", mae_1d_naive)

# computing oracle cutoff
t_grid = np.arange(
    np.min(conf_scores_1d),
    np.max(conf_scores_1d),
    0.01,
)

# computing MC integral for all t_grid
coverage_array = np.zeros(t_grid.shape[0])
for t in t_grid:
    coverage_array[t_grid == t] = np.mean(conf_scores_1d <= t)

closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
# finally, finding the naive cutoff
oracle_cutoff_1d = t_grid[closest_t_index]

# computing oracle cutoff
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

############################### Constructing illustration
# Now plotting and displaying results
# generating grid of thetas
theta = torch.linspace(-1.005, 1.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

# Compute log probabilities for both observations
log_probs_obs_2d = np.exp(
    post_estim_2d.log_prob(
        x=X_obs.to(device),
        theta=theta_grid.to(device),
    )
    .cpu()
    .numpy()
)

# Create masks for both observations
cdf_mask_obs = -log_probs_obs_2d < cdf_cutoff_2d
global_mask_obs = -log_probs_obs_2d < global_cutoff_2d
naive_mask_obs = -log_probs_obs_2d < naive_cutoff_2d
real_mask_obs = -log_probs_obs_2d < oracle_cutoff_2d

# Reshape masks for plotting
cdf_mask_obs = cdf_mask_obs.reshape(len(theta), len(theta))
global_mask_obs = global_mask_obs.reshape(len(theta), len(theta))
naive_mask_obs = naive_mask_obs.reshape(len(theta), len(theta))
real_mask_obs = real_mask_obs.reshape(len(theta), len(theta))

# Overlay all regions in a single plot for direct comparison
fig, (ax, ax_bar) = plt.subplots(
    1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
)
plt.rcParams.update({"font.size": 16})

# Barplot of MAE for each method
mae_methods = [
    ("CP4SBI-CDF", mae_2d_cdf, "blue"),
    ("Global", mae_2d_global, "green"),
    ("Naive", mae_2d_naive, "red"),
]
labels, maes, colors = zip(*mae_methods)
ax_bar.bar(labels, maes, color=colors, edgecolor="black", alpha=0.5)
ax_bar.set_ylabel("MAE (Coverage)")
ax_bar.set_title("MAE of Coverage")
ax_bar.set_ylim(0, max(maes) * 1.2)
# Make "CP4SBI-CDF" label bold by looping through the labels
for tick_label in ax_bar.get_xticklabels():
    if tick_label.get_text() == "CP4SBI-CDF":
        tick_label.set_fontweight("bold")
ax_bar.set_xticklabels(labels, rotation=20)

# Plot the oracle (real) region contour
# Plot the contours for LOCART and all other methods, including the oracle
ax.contour(
    cdf_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="blue",
    linewidths=2,
    alpha=0.75,
)
ax.contour(
    global_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="green",
    linewidths=2,
    alpha=0.75,
)
ax.contour(
    naive_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="red",
    linewidths=2,
    alpha=0.75,
)
ax.contour(
    real_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=2,
    alpha=0.75,
)

# Add legend handles manually for clarity
# Use plain text labels; color is indicated by the edgecolor of the Patch
legend_elements = [
    Patch(
        facecolor="none",
        edgecolor="blue",
        linewidth=2,
        label=r"$\mathbf{CP4SBI\text{-}CDF}$",
        alpha=0.75,
    ),
    Patch(
        facecolor="none",
        edgecolor="green",
        linewidth=2,
        label="Global",
        alpha=0.75,
    ),
    Patch(
        facecolor="none",
        edgecolor="red",
        linewidth=2,
        label="Naive",
        alpha=0.75,
    ),
    Patch(
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        label="Oracle",
        alpha=0.75,
    ),
]
# Add legend with colored text matching the edgecolor

# Create legend with colored text
legend = ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(legend_elements),
    frameon=False,
)

# Place the legend above the plot, beside the title, in a horizontal layout
ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(legend_elements),
    frameon=False,
)
ax.set_title("2D credible regions")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_xlim(-0.4, 1.0025)
ax.set_ylim(-1.025, 0.25)
plt.tight_layout()
plt.show()
# Save the current figure as PDF
fig.savefig("credible_regions_comparison_2.pdf", bbox_inches="tight")


# making confidence intervals for 1d
theta_grid_1d = torch.linspace(-1.005, 1.005, 3000)
# Compute log probabilities for both observations
log_probs_obs_1d = np.exp(
    post_estim_1d.log_prob(
        x=X_obs.to(device),
        theta=theta_grid_1d.unsqueeze(1).to(device),
    )
    .cpu()
    .numpy()
)

cdf_mask_obs_1d = -log_probs_obs_1d < cdf_cutoff_1d
global_mask_obs_1d = -log_probs_obs_1d < global_cutoff_1d
naive_mask_obs_1d = -log_probs_obs_1d < naive_cutoff_1d
real_mask_obs_1d = -log_probs_obs_1d < oracle_cutoff_1d

cdf_conf_1d = theta_grid_1d[cdf_mask_obs_1d].numpy()
global_conf_1d = theta_grid_1d[global_mask_obs_1d].numpy()
naive_conf_1d = theta_grid_1d[naive_mask_obs_1d].numpy()
real_conf_1d = theta_grid_1d[real_mask_obs_1d].numpy()

methods = [
    ("CP4SBI-CDF", cdf_conf_1d, "blue"),
    ("Global", global_conf_1d, "green"),
    ("Naive", naive_conf_1d, "red"),
]


# Plot all 1D confidence intervals
plt.figure(figsize=(8, 5))
plt.rcParams.update({"font.size": 14})

for label, conf, color in methods:
    # selecting i-th entry of conf list
    conf_value = conf
    x = np.repeat(np.array([label]), conf_value.shape[0])
    plt.errorbar(
        conf_value,
        x,
        label=label,
        color=color,
        lw=3,
    )

plt.axvline(x=np.min(real_conf_1d), linestyle="dashed", color="black")
plt.axvline(x=np.max(real_conf_1d), linestyle="dashed", color="black")

plt.xlim(-0.2, 1)
plt.title("1D Comparisson", fontweight="bold")
plt.xlabel(r"$\theta_1$")
plt.yticks([])
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
