# illustrating our approach using 2d simulators
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

# testing posterior estimators
from sbi.utils import BoxUniform
from sbi.inference import NPSE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import KDE_HPDScore
from CP4SBI.utils import naive_method
from scipy.stats import gaussian_kde

from matplotlib.lines import Line2D
from CP4SBI.gmm_task import GaussianMixture

original_path = os.getcwd()
device = "cpu"
prior_NPE = BoxUniform(
    low=-3 * torch.ones(2),
    high=3 * torch.ones(2),
    device=device,
)

############################### Deriving the cutoffs for each method
torch.manual_seed(75)
torch.cuda.manual_seed(75)

task = GaussianMixture(dim=2, prior_bound=3.0)
simulator = task.get_simulator()
prior = task.get_prior()

cuda = device == "cuda"

B = 20000
prop_calib = 0.2

B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)
theta_train = prior(num_samples=B_train)
X_train = simulator(theta_train)

# training conformal methods
thetas_calib = prior(num_samples=B_calib)
X_calib = simulator(thetas_calib)

# fitting diffusion model
inference = NPSE(prior_NPE, device=device)
inference.append_simulations(
    theta=theta_train.to(device),
    x=X_train.to(device),
).train()

# LOCART
bayes_conf = BayCon(
    sbi_score=KDE_HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    split_calib=False,
    weighting=True,
    cuda=cuda,
    alpha=0.1,
)
bayes_conf.fit(
    X=X_train,
    theta=theta_train,
)
# computing res
res = bayes_conf.locart.sbi_score.compute(X_calib, thetas_calib)

bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=res,
    min_samples_leaf=300,
    using_res=True,
)

# constructing illustration
post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)

target_coverage = 0.9
torch.manual_seed(145)
torch.cuda.manual_seed(145)
# first X_obs
theta_real = torch.tensor([[0.15, -0.1]])
# generating X_obs
X_obs = simulator(theta_real)

locart_cutoff = bayes_conf.predict_cutoff(X_obs)

# obtaining the naive cutoff
naive_cutoff = naive_method(
    post_estim,
    X=X_obs,
    alpha=0.1,
    score_type="HPD",
    device=device,
    B_naive=1000,
    kde=True,
)

# first simulating sample from the posterior
posterior_samples = post_estim.sample((1000,), x=X_obs.to(device)).cpu().numpy()

# deriving KDE estimator and contour lines
kde = gaussian_kde(posterior_samples.T)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = kde(positions).reshape(X.shape)

# obtaining the credible region using locart
# using Z already
credible_mask = -Z <= locart_cutoff
naive_mask = -Z <= naive_cutoff

# comparisson with ground truth
true_post_samples = task._sample_reference_posterior(
    num_samples=1000,
    observation=X_obs,
)

kde_true = -kde(true_post_samples.T)

t_grid = np.arange(
    np.min(kde_true),
    np.max(kde_true),
    0.001,
)

# computing MC integral for all t_grid
coverage_array = np.zeros(t_grid.shape[0])
for t in t_grid:
    coverage_array[t_grid == t] = np.mean(kde_true <= t)

closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
# finally, finding the naive cutoff
oracle_cutoff = t_grid[closest_t_index]

oracle_mask = -Z <= oracle_cutoff

# computing coverage
coverage = np.mean(kde_true <= locart_cutoff)
coverage_naive = np.mean(kde_true <= naive_cutoff)
coverage_oracle = np.mean(kde_true <= oracle_cutoff)
print(f"Coverage of the credible region: {coverage:.3f}")
print(f"Naive coverage: {coverage_naive:.3f}")

# generating the panels
fig, axs = plt.subplots(1, 3, figsize=(18, 8))
plt.rcParams.update({"font.size": 16})

# Panel 1: Scatter plot of posterior samples
axs[0].scatter(
    posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.3, s=8, color="black"
)
axs[0].set_title("Samples generated from estimated posterior")
axs[0].set_xlabel(r"$\theta_1$")
axs[0].set_ylabel(r"$\theta_2$")

# Panel 2: KDE contour lines
axs[1].contour(X, Y, Z, levels=10)
axs[1].set_title("Fitted KDE")
axs[1].set_xlabel(r"$\theta_1$")
axs[1].set_ylabel("")
# Overlay scatter plot on KDE contours
axs[1].scatter(
    posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.3, s=8, color="black"
)

# Panel 3: Credible region using locart_cutoff
# Evaluate log-probability for each grid point
# Plot oracle_mask contour (unfilled, black lines)
axs[2].contour(
    X, Y, oracle_mask, levels=[0.5], colors="black", linewidths=2, label="Oracle Cutoff"
)
# Plot naive_mask contour (unfilled, red dashed lines)
axs[2].contour(
    X,
    Y,
    naive_mask,
    levels=[0.5],
    colors="red",
    linestyles="dashed",
    linewidths=2,
    label="Naive Cutoff",
)
axs[2].contourf(
    X,
    Y,
    credible_mask,
    levels=[0.5, 1],
    colors=["tab:blue"],
    alpha=0.5,
    label="LOCART region",
)
axs[2].contour(X, Y, Z, levels=10, colors="k", linewidths=0.5, alpha=0.3)
axs[2].set_title("Credible Region")
axs[2].set_xlabel(r"$\theta_1$")
axs[2].set_ylabel("")
# Overlay scatter plot on credible region panel
axs[2].scatter(
    posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.3, s=8, color="black"
)
# Add coverage as a text element in the panel
axs[2].text(
    0.025,
    0.975,
    r"$\mathbf{CP4SBI}$ = " + f"{coverage*100:.1f}%",
    transform=axs[2].transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="tab:blue", alpha=0.5),
)

axs[2].text(
    0.395,
    0.975,
    f"Oracle = 90.0%",
    transform=axs[2].transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
)

axs[2].text(
    0.735,
    0.975,
    f"Naive = {coverage_naive*100:.1f}%",
    transform=axs[2].transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.5),
)

# Set the same axis limits for all subplots
for ax in axs:
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-3.05, 3.05)

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], color="black", lw=2, label="Oracle"),
    Line2D([0], [0], color="red", lw=2, linestyle="dashed", label="Naive"),
    Line2D(
        [0],
        [0],
        marker="s",
        color="tab:blue",
        markersize=10,
        linestyle="None",
        alpha=0.5,
        label=r"$\mathbf{CP4SBI\text{-}LOCART}$",
    ),
]

# Display the legend outside the plot
fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0015),
    ncol=4,
    frameon=True,
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()  # leave space at the top for the legend
fig.savefig("illustration_diffusion_CP4SBI.pdf", format="pdf")
