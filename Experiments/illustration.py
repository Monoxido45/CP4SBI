# illustrating our approach using 2d simulators
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

# testing posterior estimators
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from CP4SBI.utils import naive_method

# testing naive
import sbibm
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

original_path = os.getcwd()
device = "cuda"
prior_NPE = BoxUniform(
    low=-1 * torch.ones(2),
    high=1 * torch.ones(2),
    device=device,
)

############################### Deriving the cutoffs for each method
torch.manual_seed(75)
torch.cuda.manual_seed(75)
# training the posterior estimator and separating calibration samples
task_name = "two_moons"

task = sbibm.get_task(task_name)
simulator = task.get_simulator()
prior = task.get_prior()

B = 20000
prop_calib = 0.2

B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)
theta_train = prior(num_samples=B_train)
X_train = simulator(theta_train)

# training conformal methods
thetas_calib = prior(num_samples=B_calib)
X_calib = simulator(thetas_calib)

inference = NPE(prior_NPE, device=device)
inference.append_simulations(theta_train, X_train)
inference.train()

# fitting each method for obtaining the posterior credible region
# LOCART
bayes_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    split_calib=False,
    cuda=True,
    alpha=0.1,
)
bayes_conf.fit(
    X=X_train,
    theta=theta_train,
)
bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
    min_samples_leaf=100,
)

# fitting global conf
global_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="global",
    cuda=True,
    alpha=0.1,
)
global_conf.fit(
    X=X_train,
    theta=theta_train,
)

global_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
)

post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)

# Generating two X_obs, computing cutoffs and coverage for each case
target_coverage = 0.9
torch.manual_seed(45)
torch.cuda.manual_seed(45)
# first X_obs
theta_real = torch.tensor([[0.1, -0.3], [-0.3, 0.7]])
# generating X_obs
X_obs = simulator(theta_real)

# deriving cutoffs for X_obs
global_cutoff = global_conf.predict_cutoff(X_obs)
locart_cutoff = bayes_conf.predict_cutoff(X_obs)
naive_cutoff = np.zeros(X_obs.shape[0])
oracle_cutoff = np.zeros(X_obs.shape[0])
dif_cutoff_locart, dif_cutoff_global, dif_cutoff_naive = (
    np.zeros(X_obs.shape[0]),
    np.zeros(X_obs.shape[0]),
    np.zeros(X_obs.shape[0]),
)

post_samples_list = []
i = 0
for X in X_obs:
    X = X.unsqueeze(0)  # Add batch dimension
    # computing naive cutoff
    naive_cutoff[i] = naive_method(
        post_estim,
        X=X,
        alpha=0.1,
        score_type="HPD",
        device=device,
        B_naive=1000,
    )

    # ground truth
    post_samples_list.append(
        task._sample_reference_posterior(
            num_samples=2000,
            num_observation=1,
            observation=X,
        )
    )

    conf_scores = -np.exp(
        post_estim.log_prob(
            x=X.to(device),
            theta=post_samples_list[i].to(device),
        )
        .cpu()
        .numpy()
    )

    # computing oracle cutoff
    t_grid = np.arange(
        np.min(conf_scores),
        np.max(conf_scores),
        0.05,
    )

    # computing MC integral for all t_grid
    coverage_array = np.zeros(t_grid.shape[0])
    for t in t_grid:
        coverage_array[t_grid == t] = np.mean(conf_scores <= t)

    closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
    # finally, finding the naive cutoff
    oracle_cutoff[i] = t_grid[closest_t_index]

    coverage_locart = np.mean(conf_scores < locart_cutoff[i])
    coverage_global = np.mean(conf_scores < global_cutoff[i])
    coverage_naive = np.mean(conf_scores < naive_cutoff[i])

    dif_cutoff_locart[i] = np.abs(locart_cutoff[i] - oracle_cutoff[i])
    dif_cutoff_global[i] = np.abs(global_cutoff[i] - oracle_cutoff[i])
    dif_cutoff_naive[i] = np.abs(naive_cutoff[i] - oracle_cutoff[i])

    print(f"Coverage LOCART: {coverage_locart}")
    print(f"Coverage Global: {coverage_global}")
    print(f"Coverage Naive: {coverage_naive}")
    i += 1


############################### Obtaining the credible regions
# generating grid of thetas
theta1 = torch.linspace(-1, 1, 3000)
theta2 = torch.linspace(-1, 1, 3000)
theta_grid = torch.cartesian_prod(theta1, theta2)

# Compute log probabilities for both observations
log_probs_obs1 = np.exp(
    post_estim.log_prob(
        x=X_obs[0].unsqueeze(0).to(device),
        theta=theta_grid.to(device),
    )
    .cpu()
    .numpy()
)

log_probs_obs2 = np.exp(
    post_estim.log_prob(
        x=X_obs[1].unsqueeze(0).to(device),
        theta=theta_grid.to(device),
    )
    .cpu()
    .numpy()
)

# Create masks for both observations
locart_mask_obs1 = -log_probs_obs1 < locart_cutoff[0]
global_mask_obs1 = -log_probs_obs1 < global_cutoff[0]
naive_mask_obs1 = -log_probs_obs1 < naive_cutoff[0]
real_mask_obs1 = -log_probs_obs1 < oracle_cutoff[0]

locart_mask_obs2 = -log_probs_obs2 < locart_cutoff[1]
global_mask_obs2 = -log_probs_obs2 < global_cutoff[1]
naive_mask_obs2 = -log_probs_obs2 < naive_cutoff[1]
real_mask_obs2 = -log_probs_obs2 < oracle_cutoff[1]

# Reshape masks for plotting
locart_mask_obs1 = locart_mask_obs1.reshape(len(theta1), len(theta2))
global_mask_obs1 = global_mask_obs1.reshape(len(theta1), len(theta2))
naive_mask_obs1 = naive_mask_obs1.reshape(len(theta1), len(theta2))
real_mask_obs1 = real_mask_obs1.reshape(len(theta1), len(theta2))

locart_mask_obs2 = locart_mask_obs2.reshape(len(theta1), len(theta2))
global_mask_obs2 = global_mask_obs2.reshape(len(theta1), len(theta2))
naive_mask_obs2 = naive_mask_obs2.reshape(len(theta1), len(theta2))
real_mask_obs2 = real_mask_obs2.reshape(len(theta1), len(theta2))

# Update the plot layout to 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.rcParams.update({"font.size": 14})

# Add subtitles for the LOCART plots
axes[0, 0].annotate(
    "First Observation: (0.1, -0.3)",
    xy=(0.1, 1.15),
    xycoords="axes fraction",
    fontsize=18,
    fontweight="bold",
    ha="center",
    va="center",
    annotation_clip=False,
)

axes[1, 0].set_title("", loc="left")
axes[1, 0].annotate(
    "Second Observation: (-0.3, 0.7)",
    xy=(0.15, 1.15),
    xycoords="axes fraction",
    fontsize=18,
    fontweight="bold",
    ha="center",
    va="center",
    annotation_clip=False,
)

# Add a legend for the oracle region above all subplots
# LOCART region for first observation
axes[0, 0].contour(
    real_mask_obs1.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=1.5,
)
# Create a custom colormap with white as the lowest value and blue as the highest
white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
# Create custom colormaps for each method
white_green_cmap = LinearSegmentedColormap.from_list("white_green", ["white", "green"])
white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

# Use custom colormaps for the first row
axes[0, 0].imshow(
    locart_mask_obs1.T,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap=white_blue_cmap,
    alpha=0.6,
)
axes[0, 0].set_title("CP4SBI-LOCART (our approach)", fontweight="bold", color="blue")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel(r"$\theta_2$")
axes[0, 0].set_xlim(-0.025, 0.5)
axes[0, 0].set_ylim(-0.5, -0.045)

# Global region for first observation
axes[0, 1].contour(
    real_mask_obs1.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=1.5,
)

axes[0, 1].imshow(
    global_mask_obs1.T,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap=white_green_cmap,
    alpha=0.6,
)

axes[0, 1].set_title("Global CP", color="green")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("")
axes[0, 1].set_xlim(-0.025, 0.5)
axes[0, 1].set_ylim(-0.5, -0.045)

# Naive region for first observation
axes[0, 2].contour(
    real_mask_obs1.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=1.5,
)
axes[0, 2].imshow(
    naive_mask_obs1.T,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap=white_red_cmap,
    alpha=0.6,
)
axes[0, 2].set_title("Self-calibration", color="red")
axes[0, 2].set_xlabel("")
axes[0, 2].set_ylabel("")
axes[0, 2].set_xlim(-0.025, 0.5)
axes[0, 2].set_ylim(-0.5, -0.045)

# LOCART region for second observation
axes[1, 0].contour(
    real_mask_obs2.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=1.5,
)
axes[1, 0].imshow(
    locart_mask_obs2.T,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap=white_blue_cmap,
    alpha=0.6,
)
axes[1, 0].set_title("")
axes[1, 0].set_xlabel(r"$\theta_1$")
axes[1, 0].set_ylabel(r"$\theta_2$")
axes[1, 0].set_xlim(-0.95, -0.15)
axes[1, 0].set_ylim(0.2, 0.9)

# Global region for second observation
axes[1, 1].contour(
    real_mask_obs2.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=1.5,
)
axes[1, 1].imshow(
    global_mask_obs2.T,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap=white_green_cmap,
    alpha=0.6,
)
axes[1, 1].set_title("")
axes[1, 1].set_xlabel(r"$\theta_1$")
axes[1, 1].set_ylabel("")
axes[1, 1].set_xlim(-0.95, -0.15)
axes[1, 1].set_ylim(0.2, 0.9)

# Naive region for second observation
axes[1, 2].contour(
    real_mask_obs2.T,
    levels=[0.5],
    extent=(-1, 1, -1, 1),
    colors="black",
    linewidths=1.5,
)
axes[1, 2].imshow(
    naive_mask_obs2.T,
    extent=(-1, 1, -1, 1),
    origin="lower",
    cmap=white_red_cmap,
    alpha=0.6,
)
axes[1, 2].set_title("")
axes[1, 2].set_xlabel(r"$\theta_1$")
axes[1, 2].set_ylabel("")
axes[1, 2].set_xlim(-0.95, -0.15)
axes[1, 2].set_ylim(0.2, 0.9)

# Add a unified legend for the contour plot above all subplots

# Create a custom legend for the oracle region
legend_elements = [
    Line2D([0], [0], color="black", lw=1.5, label="Oracle Region"),
]

fig.subplots_adjust(top=0.85)  # Adjust the top margin to make space for the legend
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    fontsize=18,
    frameon=True,
)

plt.tight_layout()
plt.savefig(
    "illustration_locart",
    bbox_inches="tight",
)
