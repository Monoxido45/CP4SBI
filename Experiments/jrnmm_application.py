import numpy as np
from Experiments.jrnmm import simulate_jrnmm
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

from tqdm import tqdm
import gc
import pickle

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

# sbi checks for prior, simulator, and data consistency
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(model, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)

theta_0 = torch.tensor([135.0, 220.0, 2000.0]).view(1, -1)
x_0 = model(theta_0)

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

with open('Results/dens_list_jrnmm.pkl', 'wb') as f:
    pickle.dump(dens_list, f)

with open('Results/inf_list_jrnmm.pkl', 'wb') as f:
    pickle.dump(inf_list, f)