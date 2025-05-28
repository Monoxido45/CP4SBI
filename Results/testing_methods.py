import pickle
import os
import torch
import numpy as np

# testing posterior estimators
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore

# testing naive
from CP4SBI.utils import naive_method
import sbibm

original_path = os.getcwd()

###############################################################
# part of the code to check partial results
# Path to the pickle file
spec_path = "Results/MAE_results/HPD_gaussian_linear_uniform_checkpoints.pkl"

# Join the original path and file path
file_path = os.path.join(original_path, spec_path)
# Load the pickle file
with open(file_path, "rb") as file:
    slcp_checkpoint = pickle.load(file)
print(slcp_checkpoint)

###############################################################
# part of the code to debug code
torch.manual_seed(75)
torch.cuda.manual_seed(75)
task_name = "gaussian_mixture"
B = 10000
prop_calib = 0.2
B_train = int(B * (1 - prop_calib))
B_calib = int(B * prop_calib)
device = "cuda"
prior_NPE = BoxUniform(
    low=-10 * torch.ones(2),
    high=10 * torch.ones(2),
    device=device,
)


if task_name in ["sclp", "sir", "lotka_volterra", "bernoulli_glm"]:
    # Load the X_list pickle file from the X_data folder
    x_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_X_samples_10000.pkl"
    )
    with open(x_data_path, "rb") as f:
        X_list = pickle.load(f)

    # Load the X_list pickle file from the X_data folder
    theta_data_path = os.path.join(
        original_path, "Results/X_data", f"{task_name}_theta_samples_10000.pkl"
    )
    with open(theta_data_path, "rb") as f:
        theta_list = pickle.load(f)

    # exploring only the first element
    X = X_list[0]
    theta = theta_list[0]

    # splitting X
    indices = torch.randperm(X.shape[0])
    train_indices = indices[:B_train]
    calib_indices = indices[B_train:]

    X_train = X[train_indices]
    X_calib = X[calib_indices]

    # splitting theta
    theta_train = theta[train_indices]
    thetas_calib = theta[calib_indices]

else:
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()
    prior = task.get_prior()

    B_train = int(B * (1 - prop_calib))
    B_calib = int(B * prop_calib)
    theta_train = prior(num_samples=B_train)
    X_train = simulator(theta_train)

    # training conformal methods
    thetas_calib = prior(num_samples=B_calib)
    X_calib = simulator(thetas_calib)


# fitting NPE
inference = NPE(prior_NPE, device=device)
inference.append_simulations(theta_train, X_train)
inference.train()

cuda = device == "cuda"
alpha = 0.1

# A-LOCART applied
locart_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    weighting=True,
    split_calib=False,
    cuda=cuda,
    alpha=alpha,
)

locart_conf.fit(
    X=X_train,
    theta=theta_train,
)

locart_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
    min_samples_leaf=150,
)

# CDF split + LOCART
cdf_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="CDF local",
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
    min_samples_leaf=150,
)

# getting posterior data
posterior = inference.build_posterior()

posterior_data_path = (
    original_path + f"/Results/posterior_data/{task_name}_posterior_samples.pkl"
)
with open(posterior_data_path, "rb") as f:
    X_dict = pickle.load(f)

keys = list(X_dict.keys())
X_obs = torch.cat(list(X_dict.keys())).numpy()

# testing naive method
t_cutoff = naive_method(
    post_estim=posterior,
    X=X_obs,
    alpha=0.1,
)

par_n = thetas_calib.shape[0]
log_prob_array = np.zeros(par_n)
for i in range(par_n):
    log_prob_array[i] = (
        posterior.log_prob(
            thetas_calib[i].reshape(1, -1).to(device),
            x=X_calib[i].reshape(1, -1).to(device),
        )
        .cpu()
        .detach()
        .numpy()
    )

global_cutoff = np.quantile(-np.exp(log_prob_array), 1 - 0.1)

# computing cutoffs
cdf_cutoff = cdf_conf.predict_cutoff(
    X_test=X_obs,
)

locart_cutoff = locart_conf.predict_cutoff(
    X_test=X_obs,
)

# now, getting the posterior samples
for X in keys:
    true_samples = X_dict[X]
    log_probability_true_theta = -np.exp(
        posterior.log_prob(true_samples.to(device), x=X_obs.to(device)).cpu().numpy()
    )


true_samples = X_dict[X_obs]
log_probability_true_theta = -np.exp(
    posterior.log_prob(true_samples.to(device), x=X_obs.to(device)).cpu().numpy()
)

np.mean(log_probability_true_theta <= t_cutoff)
np.mean(log_probability_true_theta <= global_cutoff)
np.mean(log_probability_true_theta <= cdf_cutoff)
np.mean(log_probability_true_theta <= locart_cutoff)
