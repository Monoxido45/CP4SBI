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
torch.manual_seed(45)
torch.cuda.manual_seed(45)
# first X_obs
theta_real = torch.tensor([[1.25, -0.3], [0.75, 0.7]])
# generating X_obs
X_obs = simulator(theta_real)

locart_cutoff = bayes_conf.predict_cutoff(X_obs)

# generating the panels
