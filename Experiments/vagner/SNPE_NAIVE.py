import torch
from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi, SNPE_C
from sbi.utils import BoxUniform
from CP4SBI.utils import naive_method, hdr_method
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore, WALDOScore
import sbibm
from copy import deepcopy
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
import numpy as np

num_rounds = 10
alpha = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
score_type = "HPD"
task_name = "gaussian_linear"
naive_samples = 1000
B_train = 100
B_calib = 100

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

if task_name != "gaussian_mixture":
    task = sbibm.get_task(task_name)
    original_simulator = task.get_simulator()
    prior = task.get_prior()
else:
    from CP4SBI.gmm_task import GaussianMixture
    task = GaussianMixture(dim=2, prior_bound=4.0)
    original_simulator = task.get_simulator()
    prior = task.get_prior()

if task_name == "gaussian_linear":
    prior_params = {
        "loc": torch.zeros((task.dim_parameters,), device=device),
        "precision_matrix": torch.inverse(0.1 * torch.eye(task.dim_parameters, device=device)),
    }
    prior_dist = MultivariateNormal(**prior_params, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
else:
    prior_NPE, _, _ = process_prior(prior)

prior, num_parameters, prior_returns_numpy = process_prior(prior_dist if task_name == "gaussian_linear" else prior)

def device_simulator(theta):
    theta = theta.cpu()
    return original_simulator(theta).to(device)  
simulator = device_simulator

#check_sbi_inputs(device_simulator, prior)

theta_train = prior.sample((B_train,)).to(device)
X_train = simulator(theta_train).to(device)

thetas_calib = prior.sample((B_calib,)).to(device)
X_calib = simulator(thetas_calib).to(device)

inference = SNPE_C(prior=prior_NPE, device=device)

posteriors = []
proposal = prior

theta_o = prior.sample((1,)).to(device)
x_o = simulator(theta_o).to(device)
X_0 = x_o

for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

post_estim = posteriors[-1]
post_samples = post_estim.sample((1000,), x=x_o.to(device))

if score_type == "HPD":
    hdr_cutoff, hdr_obj = hdr_method(
        post_estim=inference,
        X_calib=X_calib,
        thetas_calib=thetas_calib,
        n_grid=1000,
        X_test=X_0,
        is_fitted=True,
        alpha=alpha,
        score_type=score_type,
        device=device,
        post_dens=density_estimator,
    )

    closest_t = naive_method(
        post_estim,
        X=X_0,
        alpha=alpha,
        score_type=score_type,
        device=device,
        n_grid=1000,
        B_naive=naive_samples,
    )

    conf_scores = -np.exp(post_estim.log_prob(post_samples.to(device=device), x=X_0.to(device=device)).cpu()).cpu().numpy()

    _, dens_samples = hdr_obj.recal_sample(
        y_hat=post_samples.cpu().reshape(1, post_samples.shape[0], post_samples.shape[1]),
        f_hat_y_hat=-conf_scores.reshape(1, -1),
    )

    hdr_conf_scores = -dens_samples[0, :]

elif score_type == "WALDO":
    closest_t, mean_array, inv_matrix = naive_method(
        post_estim,
        X=X_0,
        alpha=alpha,
        score_type=score_type,
        device=device,
        B_naive=naive_samples,
    )

    conf_scores = np.zeros(post_samples.shape[0])
    for j in range(post_samples.shape[0]):
        if mean_array.shape[0] > 1:
            sel_sample = post_samples[j, :].cpu().numpy()
            conf_scores[j] = (mean_array - sel_sample).transpose() @ inv_matrix @ (mean_array - sel_sample)
        else:
            sel_sample = post_samples[j].cpu().numpy()
            conf_scores[j] = (mean_array - sel_sample) ** 2 / (inv_matrix)

print(np.mean(conf_scores <= closest_t))




import torch
from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi, SNPE_C
from sbi.utils import BoxUniform
from CP4SBI.utils import naive_method, hdr_method
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore, WALDOScore
import sbibm
from copy import deepcopy
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
import numpy as np

num_rounds = 10
alpha = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
score_type = "HPD"
task_name = "gaussian_linear"
naive_samples = 1000
B_train = 100
B_calib = 100
num_test = 100

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

if task_name != "gaussian_mixture":
    task = sbibm.get_task(task_name)
    original_simulator = task.get_simulator()
    prior = task.get_prior()
else:
    from CP4SBI.gmm_task import GaussianMixture
    task = GaussianMixture(dim=2, prior_bound=4.0)
    original_simulator = task.get_simulator()
    prior = task.get_prior()

if task_name == "gaussian_linear":
    prior_params = {
        "loc": torch.zeros((task.dim_parameters,), device=device),
        "precision_matrix": torch.inverse(0.1 * torch.eye(task.dim_parameters, device=device)),
    }
    prior_dist = MultivariateNormal(**prior_params, validate_args=False)
    prior_NPE, _, _ = process_prior(prior_dist)
else:
    prior_NPE, _, _ = process_prior(prior)

prior, num_parameters, prior_returns_numpy = process_prior(prior_dist if task_name == "gaussian_linear" else prior)

def device_simulator(theta):
    theta = theta.cpu()
    return original_simulator(theta).to(device)  
simulator = device_simulator

theta_train = prior.sample((B_train,)).to(device)
X_train = simulator(theta_train).to(device)

thetas_calib = prior.sample((B_calib,)).to(device)
X_calib = simulator(thetas_calib).to(device)

theta_test = prior.sample((num_test,)).to(device)
X_test = simulator(theta_test).to(device)

coverages = []

for i in range(num_test):
    x_o = X_test[i:i+1]
    theta_o = theta_test[i:i+1]
    
    inference = SNPE_C(prior=prior_NPE, device=device)
    posteriors = []
    proposal = prior

    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)

    post_estim = posteriors[-1]
    post_samples = post_estim.sample((1000,), x=x_o.to(device))

    if score_type == "HPD":
        hdr_cutoff, hdr_obj = hdr_method(
            post_estim=inference,
            X_calib=X_calib,
            thetas_calib=thetas_calib,
            n_grid=1000,
            X_test=x_o,
            is_fitted=True,
            alpha=alpha,
            score_type=score_type,
            device=device,
            post_dens=density_estimator,
        )

        closest_t = naive_method(
            post_estim,
            X=x_o,
            alpha=alpha,
            score_type=score_type,
            device=device,
            n_grid=1000,
            B_naive=naive_samples,
        )

        conf_scores = -np.exp(post_estim.log_prob(post_samples.to(device=device), x=x_o.to(device=device)).cpu()).cpu().numpy()

        _, dens_samples = hdr_obj.recal_sample(
            y_hat=post_samples.cpu().reshape(1, post_samples.shape[0], post_samples.shape[1]),
            f_hat_y_hat=-conf_scores.reshape(1, -1),
        )

        hdr_conf_scores = -dens_samples[0, :]
        coverage = np.mean(conf_scores <= closest_t)
        coverages.append(coverage)

    elif score_type == "WALDO":
        closest_t, mean_array, inv_matrix = naive_method(
            post_estim,
            X=x_o,
            alpha=alpha,
            score_type=score_type,
            device=device,
            B_naive=naive_samples,
        )

        conf_scores = np.zeros(post_samples.shape[0])
        for j in range(post_samples.shape[0]):
            if mean_array.shape[0] > 1:
                sel_sample = post_samples[j, :].cpu().numpy()
                conf_scores[j] = (mean_array - sel_sample).transpose() @ inv_matrix @ (mean_array - sel_sample)
            else:
                sel_sample = post_samples[j].cpu().numpy()
                conf_scores[j] = (mean_array - sel_sample) ** 2 / (inv_matrix)
        
        coverage = np.mean(conf_scores <= closest_t)
        coverages.append(coverage)

average_coverage = np.mean(coverages)
print(average_coverage)