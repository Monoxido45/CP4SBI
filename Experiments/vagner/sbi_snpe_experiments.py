import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sbi.inference import SNPE
from sbi.utils import process_prior

# for benchmarking
import sbibm
from copy import deepcopy
import pandas as pd


print("Fitting II - example .............")
# All appears to be good, now we will test and compare to the original HPD and
# Global conformal
############################ Testing our framework on two moons simulator
# Working with one of the SBI benchmarks examples
# Using i.i.d dataset
alpha = 0.1
torch.manual_seed(0)
torch.cuda.manual_seed(0)
task = sbibm.get_task("two_moons")
n_samples = 500

prior = task.get_prior()
simulator = task.get_simulator()

prior_NPE = BoxUniform(low=-1 * torch.ones(2), high=1 * torch.ones(2), device="cuda")

# get one observation in particular
observation = task.get_observation(num_observation=1)
post_samples = task.get_reference_posterior_samples(num_observation=1)

post_samples_2 = task._sample_reference_posterior(
    num_samples=500, num_observation=20, observation=observation
)

# training NPE with few samples
thetas = prior(num_samples=n_samples)
X_train = simulator(thetas)

# specifying model
inference = NPE(prior_NPE, device="cuda")
inference.append_simulations(thetas, X_train).train()

# training conformal methods with additional samples
calib_samples = 200
thetas_calib = prior(num_samples=calib_samples)
X_calib = simulator(thetas_calib)

# LOCART
bayes_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    cuda=True,
    alpha=alpha,
)
bayes_conf.fit(
    X=X_train,
    theta=thetas,
)

bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
    locart_kwargs={"min_samples_leaf": 100},
)

# global
global_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="global",
    cuda=True,
    alpha=alpha,
)

global_conf.fit(
    X=X_train,
    theta=thetas,
)

global_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
)

# Naive HPD for comparison
# generating from posterior
naive_samples = 1000
posterior = deepcopy(bayes_conf.locart.sbi_score.posterior)
samples = posterior.sample(
    (naive_samples,),
    x=observation.to(device="cuda"),
)

conf_scores = -np.exp(
    posterior.log_prob_batched(samples, x=observation.to(device="cuda")).cpu().numpy()
)

# picking large grid between maximum and minimum densities
t_grid = np.arange(np.min(conf_scores), np.max(conf_scores), 0.005)
target_coverage = 1 - alpha

# computing MC integral for all t_grid
coverage_array = np.zeros(t_grid.shape[0])
for t in t_grid:
    coverage_array[t_grid == t] = np.mean(conf_scores <= t)

closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
closest_t = t_grid[closest_t_index]


# computing coverage for each approach
# computing density for each posterior observation
dens_test = -np.exp(
    posterior.log_prob_batched(
        post_samples.to(device="cuda"), x=observation.to(device="cuda")
    )
    .cpu()
    .numpy()
)

# computing cutoffs
locart_cutoff = bayes_conf.predict_cutoff(observation.numpy())
global_cutoff = global_conf.predict_cutoff(observation.numpy())

# computing coverage
locart_coverage = np.mean(dens_test <= locart_cutoff)
global_coverage = np.mean(dens_test <= global_cutoff)
naive_coverage = np.mean(dens_test <= closest_t)

print(f"LOCART Coverage: {locart_coverage}")
print(f"Global Coverage: {global_coverage}")
print(f"Naive Coverage: {naive_coverage}")


print("Fitting II - example with SNPE .............")
# All appears to be good, now we will test and compare to the original HPD and
# Global conformal
############################ Testing our framework on two moons simulator
# Working with one of the SBI benchmarks examples
# Using i.i.d dataset
alpha = 0.1
torch.manual_seed(0)
torch.cuda.manual_seed(0)
task = sbibm.get_task("two_moons")
n_samples = 500
num_rounds = 10

prior = task.get_prior()
simulator = task.get_simulator()

prior_NPE = BoxUniform(low=-1 * torch.ones(2), high=1 * torch.ones(2), device="cuda")

# get one observation in particular
observation = task.get_observation(num_observation=1)
post_samples = task.get_reference_posterior_samples(num_observation=1)

post_samples_2 = task._sample_reference_posterior(
    num_samples=500, num_observation=20, observation=observation
)

# training NPE with few samples
thetas = prior(num_samples=n_samples)
X_train = simulator(thetas)

# specifying model
inference = NPE(prior_NPE, device="cuda")
inference.append_simulations(thetas, X_train).train()

# training conformal methods with additional samples
calib_samples = 200
thetas_calib = prior(num_samples=calib_samples)
X_calib = simulator(thetas_calib)

# LOCART
bayes_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
    cuda=True,
    alpha=alpha,
)
bayes_conf.fit(
    X=X_train,
    theta=thetas,
)


bayes_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
    locart_kwargs={"min_samples_leaf": 100},
)

# global
global_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="global",
    cuda=True,
    alpha=alpha,
)

global_conf.fit(
    X=X_train,
    theta=thetas,
)

global_conf.calib(
    X_calib=X_calib,
    theta_calib=thetas_calib,
)

# Naive HPD for comparison
# generating from posterior
naive_samples = 1000

posteriors = []
proposal = prior_NPE
x_o = observation.to(device="cuda")
posterior = bayes_conf.locart.sbi_score.posterior
posteriors.append(posterior)
posterior = bayes_conf.locart.sbi_score.posterior
inference = bayes_conf.locart.sbi_score.inference_obj
print(x_o)

def cpu_simulator(parameters):
    with torch.no_grad():
        return simulator(parameters.cpu())
    
for i in range(num_rounds):
    
    theta, x = simulate_for_sbi(cpu_simulator, proposal, num_simulations=500)
    theta, x = theta.to("cuda"), x.to("cuda")
    
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal
    ).train(force_first_round_loss=True) # the result of SNPE will not be the true posterior
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)
    bayes_conf.locart.sbi_score.posterior = posterior

posterior = deepcopy(bayes_conf.locart.sbi_score.posterior)
samples = posterior.sample(
    (naive_samples,),
    x=observation.to(device="cuda"),
)

conf_scores = -np.exp(
    posterior.log_prob_batched(samples, x=observation.to(device="cuda")).cpu().numpy()
)

# picking large grid between maximum and minimum densities
t_grid = np.arange(np.min(conf_scores), np.max(conf_scores), 0.005)
target_coverage = 1 - alpha

# computing MC integral for all t_grid
coverage_array = np.zeros(t_grid.shape[0])
for t in t_grid:
    coverage_array[t_grid == t] = np.mean(conf_scores <= t)

closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
closest_t = t_grid[closest_t_index]


# computing coverage for each approach
# computing density for each posterior observation
dens_test = -np.exp(
    posterior.log_prob_batched(
        post_samples.to(device="cuda"), x=observation.to(device="cuda")
    )
    .cpu()
    .numpy()
)

# computing cutoffs
locart_cutoff = bayes_conf.predict_cutoff(observation.numpy())
global_cutoff = global_conf.predict_cutoff(observation.numpy())

# computing coverage
locart_coverage = np.mean(dens_test <= locart_cutoff)
global_coverage = np.mean(dens_test <= global_cutoff)
naive_coverage = np.mean(dens_test <= closest_t)

print(f"LOCART Coverage: {locart_coverage}")
print(f"Global Coverage: {global_coverage}")
print(f"Naive Coverage: {naive_coverage}")

###### Computing coverage for each approach across several X
# setting B = 5000
def compute_coverage(
    prior_NPE,
    B=5000,
    prop_calib=0.2,
    alpha=0.1,
    num_obs=500,
    task="two_moons",
    device="cuda",
    random_seed=0,
    min_samples_leaf=300,
    naive_samples=1000,
    num_p_samples=1000,
    sequential = False,
    num_rounds = 0,
):
    # fixing task
    task = sbibm.get_task(task)
    prior = task.get_prior()
    simulator = task.get_simulator()

    # setting seet
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # simulating random observations for computing coverage
    thetas_obs = prior(num_samples=num_obs)
    X_obs = simulator(thetas_obs)

    # splitting simulation budget
    B_train = int(B * (1 - prop_calib))
    B_calib = int(B * prop_calib)

    # training samples
    theta_train = prior(num_samples=B_train)
    X_train = simulator(theta_train)

    # fitting NPE
    inference = NPE(prior_NPE, device=device)
    inference.append_simulations(theta_train, X_train).train()

    # training conformal methods
    thetas_calib = prior(num_samples=B_calib)
    X_calib = simulator(thetas_calib)
    cuda = device == "cuda"

    # CDF split
    print("Fitting CDF split")
    cdf_conf = BayCon(
        sbi_score=HPDScore,
        base_inference=inference,
        is_fitted=True,
        conformal_method="CDF",
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
    )

    # fitting LOCART
    print("Fitting LOCART")
    bayes_conf = BayCon(
        sbi_score=HPDScore,
        base_inference=inference,
        is_fitted=True,
        conformal_method="local",
        cuda=cuda,
        alpha=alpha,
    )
    bayes_conf.fit(
        X=X_train,
        theta=theta_train,
    )

    bayes_conf.calib(
        X_calib=X_calib,
        theta_calib=thetas_calib,
        locart_kwargs={"min_samples_leaf": min_samples_leaf},
    )

    # global
    print("Fitting global conformal")
    global_conf = BayCon(
        sbi_score=HPDScore,
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
        theta_calib=thetas_calib,
    )

    coverage_locart = np.zeros(X_obs.shape[0])
    coverage_global = np.zeros(X_obs.shape[0])
    coverage_cdf = np.zeros(X_obs.shape[0])
    coverage_naive = np.zeros(X_obs.shape[0])

    locart_cutoff = bayes_conf.predict_cutoff(X_obs.numpy())
    global_cutoff = global_conf.predict_cutoff(X_obs.numpy())
    cdf_cutoff = cdf_conf.predict_cutoff(X_obs.numpy())

    i = 0
    # evaluating cutoff for each observation
    for X in tqdm(X_obs, desc="Computing coverage across observations"):
        post_samples = task._sample_reference_posterior(
            num_samples=num_p_samples,
            num_observation=i,
            observation=X.reshape(1, -1),
        )

        # computing naive HPD cutoff
        if (sequential):
            posteriors = []
            proposal = prior_NPE
            x_o = observation.to(device="cuda")
            posterior = bayes_conf.locart.sbi_score.posterior
            posteriors.append(posterior)
            posterior = bayes_conf.locart.sbi_score.posterior
            inference = bayes_conf.locart.sbi_score.inference_obj
            #print(x_o)

            def cpu_simulator(parameters):
                with torch.no_grad():
                    return simulator(parameters.cpu())
    
            for j in range(num_rounds):
    
                theta, x = simulate_for_sbi(cpu_simulator, proposal, num_simulations=500)
                theta, x = theta.to("cuda"), x.to("cuda")
    
                density_estimator = inference.append_simulations(
                    theta, x, proposal=proposal
                ).train(force_first_round_loss=True) # the result of SNPE will not be the true posterior
                posterior = inference.build_posterior(density_estimator)
                posteriors.append(posterior)
                proposal = posterior.set_default_x(x_o)
                bayes_conf.locart.sbi_score.posterior = posterior
            
        post_estim = deepcopy(bayes_conf.locart.sbi_score.posterior)

        samples = post_estim.sample(
            (naive_samples,),
            x=X.reshape(1, -1).to(device="cuda"),
            show_progress_bars=False,
        )

        conf_scores = -np.exp(
            post_estim.log_prob_batched(
                samples,
                x=X.reshape(1, -1).to(device="cuda"),
            )
            .cpu()
            .numpy()
        )

        # picking large grid between maximum and minimum densities
        t_grid = np.arange(
            np.min(conf_scores),
            np.max(conf_scores),
            0.005,
        )
        target_coverage = 1 - alpha

        # computing MC integral for all t_grid
        coverage_array = np.zeros(t_grid.shape[0])
        for t in t_grid:
            coverage_array[t_grid == t] = np.mean(conf_scores <= t)

        closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
        # finally, finding the naive cutoff
        closest_t = t_grid[closest_t_index]

        # computing density for each posterior observation
        dens_test = -np.exp(
            post_estim.log_prob_batched(
                post_samples.to(device=device),
                x=X.reshape(1, -1).to(device=device),
            )
            .cpu()
            .numpy()
        )

        # computing coverage
        coverage_locart[i] = np.mean(dens_test <= locart_cutoff[i])
        coverage_global[i] = np.mean(dens_test <= global_cutoff[i])
        coverage_naive[i] = np.mean(dens_test <= closest_t)
        coverage_cdf[i] = np.mean(dens_test <= cdf_cutoff[i])

        i += 1

    # Creating a pandas DataFrame with the mean coverage values
    coverage_df = pd.DataFrame(
        {
            "LOCART MAD": [np.mean(np.abs(coverage_locart - (1 - alpha)))],
            "Global CP MAD": [np.mean(np.abs(coverage_global - (1 - alpha)))],
            "Naive MAD": [np.mean(np.abs(coverage_naive - (1 - alpha)))],
            "CDF MAD": [np.mean(np.abs(coverage_cdf - (1 - alpha)))],
        }
    )
    return coverage_df


# defining prior for NPE
prior_NPE = BoxUniform(low=-1 * torch.ones(2), high=1 * torch.ones(2), device="cuda")

n_obs = 10

# Running the function
coverage_df = compute_coverage(
    prior_NPE,
    alpha=0.1,
    num_obs=n_obs,
    task="two_moons",
    device="cuda",
    random_seed=50,
    min_samples_leaf=300,
    naive_samples=1000,
)

print(coverage_df)
coverage_df.to_csv('coverage_df.csv', index=False, sep=';', quotechar='"')

coverage_df_snpe = compute_coverage(
    prior_NPE,
    alpha=0.1,
    num_obs=n_obs,
    task="two_moons",
    device="cuda",
    random_seed=50,
    min_samples_leaf=300,
    naive_samples=1000,
    sequential=True,
    num_rounds = 15,
)

print(coverage_df_snpe)
coverage_df_snpe.to_csv('coverage_df_snpe.csv', index=False, sep=';', quotechar='"')
