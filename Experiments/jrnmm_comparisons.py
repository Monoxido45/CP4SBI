# Code for comparing our method to naive, global in JRNMM experiment
import numpy as np
from Experiments.jrnmm import simulate_jrnmm
from CP4SBI.utils import naive_method
import torch
from scipy.signal import welch
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from Experiments.utils import train_sbi_amortized
import matplotlib.pyplot as plt

from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from sbi.inference import simulate_for_sbi

from tqdm import tqdm
import gc
import pickle
from copy import deepcopy
from matplotlib.patches import Patch
import os

# setting seeds for reproducibility
torch.manual_seed(75)
torch.cuda.manual_seed(75)

def model(theta):
    theta = theta.numpy()
    x = []
    for thetai in theta:
        # choose values of the JRNMM for the simulation
        C, mu, sigma = thetai
        # define timespan
        delta = 1/2**10
        burnin = 2  # given in seconds
        duration = 8  # given in seconds
        downsample = 8
        tarray = np.arange(0, burnin + duration, step=delta)
        # simulate JRNMM model with Strang splitting
        si, _ = simulate_jrnmm(mu, sigma, C, tarray, burnin, downsample)
        si = si - np.mean(si)
        _, pyyi = welch(si, nperseg=64)
        logpyyi = np.log10(pyyi)
        x.append(logpyyi)
    return torch.tensor(np.array(x))

prior = BoxUniform(
    low=torch.tensor([10.0, 50.0, 100.0]),
    high=torch.tensor([250.0, 500.0, 5000.0])
)

# defining each grid
theta_1 = torch.linspace(100, 200.0, 3000)
theta_2 = torch.linspace(100, 320.0, 3000)
theta_3 = torch.linspace(1000, 3000.0, 3000)
theta_len = 3000

# making a grid list
theta_grid_list = [torch.cartesian_prod(theta_1, theta_2),
                   torch.cartesian_prod(theta_1, theta_3),
                   torch.cartesian_prod(theta_2, theta_3)]

# sbi checks for prior, simulator, and data consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(model, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)

theta_0 = torch.tensor([135.0, 220.0, 2000.0]).view(1, -1)
x_0 = model(theta_0)

# running each nuisance parameter amortized inference
# run this chunck only once to save the density and inference list
sim_budget = 10_000
dens_list, inf_list = train_sbi_amortized(
    sim_budget=sim_budget,
    simulator=simulator,
    prior=prior,
    density_estimator='nsf',
    save_fname='Results/jrnmm_amortized',
    return_density=True,
    nuisance=True,
)

with open('Experiments/dens_list_jrnmm.pkl', 'wb') as f:
    pickle.dump(dens_list, f)

with open('Experiments/inf_list_jrnmm.pkl', 'wb') as f:
    pickle.dump(inf_list, f)

# running CP4SBI regions now
with open('Experiments/dens_list_jrnmm.pkl', 'rb') as f:
    dens_list = pickle.load(f)

with open('Experiments/inf_list_jrnmm.pkl', 'rb') as f:
    inf_list = pickle.load(f)


# calibration
cal_budget = 4000
min_samples_leaf = 300
# combination list for nuisance parameters
comb_list = [[0, 1], [0, 2], [1, 2]]
device = "cpu"
locart_masks, naive_masks, global_masks = [], [], []


# simulating samples for calibration
theta_calib, X_calib = simulate_for_sbi(
    simulator, proposal=prior, num_simulations=cal_budget
)

# after simulating, running each method for each nuisance parameter combination
for i, comb in enumerate(tqdm(comb_list)):
    inference = inf_list[i]
    dens = dens_list[i]
    thetas_calib_used = theta_calib[:, comb]

    # defining the BayCon locart object
    locart_obj = BayCon(
        sbi_score=HPDScore,
        base_inference=inference,
        density = dens,
        is_fitted=True,
        conformal_method="local",
        split_calib=False,
        weighting=True,
        cuda=device == "cuda",
        alpha=0.1,
    )
    
    locart_obj.fit(
        X=None,
        theta=None,
    )

    global_obj = BayCon(
        sbi_score=HPDScore,
        base_inference=inference,
        density = dens,
        is_fitted=True,
        conformal_method="global",
        cuda=device == "cuda",
        alpha=0.1,
    )
    global_obj.fit(
        X=None,
        theta=None,
    )
    
    # calibration step
    res = locart_obj.locart.sbi_score.compute(X_calib, thetas_calib_used)

    # deriving cutoffs
    locart_obj.calib(
        X_calib=X_calib,
        theta_calib=res,
        min_samples_leaf=min_samples_leaf,
        using_res=True,
    )

    global_obj.calib(
        X_calib=X_calib,
        theta_calib=res,
        using_res=True,
    )

    post_estim_2d = deepcopy(locart_obj.locart.sbi_score.posterior)
    
    naive_cutoff_2d = naive_method(
    post_estim_2d,
    X=x_0,
    alpha=0.1,
    score_type="HPD",
    device=device,
    B_naive=1000,
    n_grid=500,
    )

    # obtaining all cutoffs
    locart_cutoff_2d = locart_obj.predict_cutoff(x_0)
    global_cutoff_2d = global_obj.predict_cutoff(x_0)


    log_probs_obs_2d = np.exp(
        post_estim_2d.log_prob(
            x=x_0.to(device),
            theta=theta_grid_list[i].to(device),
        )
        .cpu()
        .numpy()
    )

    locart_mask_obs = -log_probs_obs_2d < locart_cutoff_2d
    global_mask_obs = -log_probs_obs_2d < global_cutoff_2d
    naive_mask_obs = -log_probs_obs_2d < naive_cutoff_2d

    locart_mask_obs = locart_mask_obs.reshape(theta_len, theta_len)
    naive_mask_obs = naive_mask_obs.reshape(theta_len, theta_len)
    global_mask_obs = global_mask_obs.reshape(theta_len, theta_len)

    locart_masks.append(locart_mask_obs)
    naive_masks.append(naive_mask_obs)
    global_masks.append(global_mask_obs)

    # clearing the memory
    del locart_obj
    del global_obj
    del naive_cutoff_2d
    del global_cutoff_2d
    del naive_mask_obs
    del global_mask_obs
    del locart_mask_obs
    del post_estim_2d
    del locart_cutoff_2d
    del res
    
    gc.collect()
    torch.cuda.empty_cache()

# ensure output directory exists and persist masks
os.makedirs("Experiments", exist_ok=True)

with open("Experiments/locart_masks_jrnmm_application.pkl", "wb") as f:
    pickle.dump(np.array(locart_masks), f)

with open("Experiments/naive_masks_jrnmm_application.pkl", "wb") as f:
    pickle.dump(np.array(naive_masks), f)

with open("Experiments/global_masks_jrnmm_application.pkl", "wb") as f:
    pickle.dump(np.array(global_masks), f)

# Plot credible region contours for each nuisance-parameter pair (3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.rcParams.update({"font.size": 14})

param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
comb_list = [[0, 1], [0, 2], [1, 2]]  # same order used above

theta_lims = [[120, 150], [150, 275], [1800, 2300]]

# Ensure theta vectors are numpy arrays for extents
theta_1_np = theta_1.numpy()
theta_2_np = theta_2.numpy()
theta_3_np = theta_3.numpy()
theta_vecs = [theta_1_np, theta_2_np, theta_3_np]

# Colors to keep consistent
colors = {"locart": "blue", "global": "green", "naive": "red"}

for i, comb in enumerate(comb_list):
    ax = axes[i]
    # pick masks for this combination (already stored in lists in same order)
    locart_mask = locart_masks[i]
    global_mask = global_masks[i]
    naive_mask = naive_masks[i]

    # extents: (xmin, xmax, ymin, ymax) from the corresponding theta vectors
    x_vec = theta_vecs[comb[0]]
    y_vec = theta_vecs[comb[1]]
    extent = (float(x_vec[0]), float(x_vec[-1]), float(y_vec[0]), float(y_vec[-1]))

    # Plot contour boundaries (level 0.5 on boolean masks) for each method
    ax.contour(
        locart_mask.T,
        levels=[0.5],
        extent=extent,
        colors=colors["locart"],
        linewidths=2,
        alpha=0.85,
    )
    ax.contour(
        global_mask.T,
        levels=[0.5],
        extent=extent,
        colors=colors["global"],
        linewidths=2,
        alpha=0.85,
    )
    ax.contour(
        naive_mask.T,
        levels=[0.5],
        extent=extent,
        colors=colors["naive"],
        linewidths=2,
        alpha=0.85,
    )

    # Labels and title for this column
    ax.set_xlabel(param_names[comb[0]])
    ax.set_ylabel(param_names[comb[1]])
    ax.set_title(f"{param_names[comb[0]]} vs {param_names[comb[1]]}")
    ax.set_xlim(extent[0], extent[1])
    # apply the fixed theta limits for the plotted parameter pair
    xlim = theta_lims[comb[0]]
    ylim = theta_lims[comb[1]]
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))

# Create legend handles and place a single legend above the subplots
legend_elements = [
    Patch(facecolor="none", edgecolor=colors["locart"], linewidth=2, label=r"$\mathbf{CP4SBI\text{-}LOCART}$"),
    Patch(facecolor="none", edgecolor=colors["global"], linewidth=2, label="Global"),
    Patch(facecolor="none", edgecolor=colors["naive"], linewidth=2, label="Self-calibration"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=len(legend_elements), frameon=False, bbox_to_anchor=(0.5, 1.03))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
fig.savefig("credible_regions_comparison_pairs.pdf", bbox_inches="tight")
