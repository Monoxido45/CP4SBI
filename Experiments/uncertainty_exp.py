import numpy as np
import os
import torch

from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from tqdm import tqdm
import sbibm
from copy import deepcopy
from matplotlib.patches import Patch
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
cdf_cutoff_2d = cdf_conf_2d.predict_cutoff(X_obs)

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

mean_coverage_2d_cdf = np.mean(conf_scores_2d <= cdf_cutoff_2d)
mae_2d_cdf = np.abs(mean_coverage_2d_cdf - target_coverage)

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

# Now plotting and displaying results
# generating grid of thetas
theta = torch.linspace(-1.005, 1.005, 3000)
theta_grid = torch.cartesian_prod(theta, theta)

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
locart_unc = locart_unc.reshape(len(theta), len(theta))
cdf_unc = cdf_unc.reshape(len(theta), len(theta))


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

cdf_mask_obs = cdf_mask_obs.reshape(len(theta), len(theta))
locart_mask_obs = locart_mask_obs.reshape(len(theta), len(theta))
real_mask_obs = real_mask_obs.reshape(len(theta), len(theta))

fig, (ax_bar, ax) = plt.subplots(1, 2, figsize=(16, 8))
plt.rcParams.update({"font.size": 16})

# Barplot of MAE for each method
mae_methods = [
    ("CP4SBI-CDF", mae_2d_cdf, "blue"),
    ("CP4SBI-LOCART", mae_2d_locart, "dodgerblue"),
]

# Plot the CDF region as before
ax_bar.contour(
    cdf_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="dodgerblue",
    linewidths=2,
    alpha=0.8,
)

ax_bar.contourf(
    cdf_unc.T,
    levels=[0.99, 1.01],
    extent=(-1, 1, -1, 1),
    colors="darkblue",
    linewidths=2,
    alpha=0.5,
)

# Shade regions where locart_unc.T == 0.5
ax_bar.contourf(
    cdf_unc.T,
    levels=[0.49, 0.51],
    extent=(-1, 1, -1, 1),
    colors="lightskyblue",
    alpha=0.5,
)

ax_bar.contour(
    real_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=2,
    alpha=0.75,
)

ax_bar.set_title("CDF uncertainty region")
ax_bar.set_xlabel(r"$\theta_1$")
ax_bar.set_ylabel(r"$\theta_2$")
ax_bar.set_ylim(-1.05, 0.3)
ax_bar.set_xlim(-0.5, 1.00)

# Plot the oracle (real) region contour

# Plot regions where locart_unc.T == 1 (boundary) and shade regions where locart_unc.T == 0.5
# First, plot the contour for locart_unc.T == 1
ax.contour(
    locart_mask_obs.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="blue",
    linewidths=2,
    alpha=0.8,
)


ax.contourf(
    locart_unc.T,
    levels=[0.99, 1.01],
    extent=(-1, 1, -1, 1),
    colors="darkblue",
    linewidths=2,
    alpha=0.5,
)

# Shade regions where locart_unc.T == 0.5
ax.contourf(
    locart_unc.T,
    levels=[0.49, 0.51],
    extent=(-1, 1, -1, 1),
    colors="lightskyblue",
    alpha=0.5,
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
        edgecolor="dodgerblue",
        linewidth=2,
        label=r"$\mathbf{CP4SBI\text{-}CDF}$",
        alpha=0.75,
    ),
    Patch(
        facecolor="none",
        edgecolor="blue",
        linewidth=2,
        label=r"$\mathbf{CP4SBI\text{-}LOCART}$",
        alpha=0.75,
    ),
    Patch(
        facecolor="darkblue",
        edgecolor="none",
        linewidth=2,
        label="Inside region",
        alpha=0.5,
    ),
    Patch(
        facecolor="lightskyblue",
        edgecolor="none",
        linewidth=2,
        label="Underterminate region",
        alpha=0.5,
    ),
    Patch(
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        label="Oracle region",
        alpha=0.75,
    ),
]
# Add legend with colored text matching the edgecolor

# Create legend with colored text
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=len(legend_elements),
    frameon=False,
)

# legend = ax.legend(
#    handles=legend_elements,
#    loc="upper center",
#    bbox_to_anchor=(0.5, 1.15),
#    ncol=len(legend_elements),
#    frameon=False,
# )
# Place the legend above the plot, beside the title, in a horizontal layout
# ax.legend(
#    handles=legend_elements,
#    loc="upper center",
#    bbox_to_anchor=(0.2, 1.15),
#    ncol=len(legend_elements),
#    frameon=False,
# )

ax.set_title("LOCART uncertainty region")
ax.set_xlabel(r"$\theta_1$")
ax.set_ylim(-1.05, 0.3)
ax.set_xlim(-0.5, 1.00)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
