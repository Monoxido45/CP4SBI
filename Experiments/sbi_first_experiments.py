import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayCon
from CP4SBI.scores import HPDScore

# Example usage:
num_dims = 3
num_sims = 1000
num_calib = 1000  # Number of calibration points

# 1. Define prior and simulator
prior = BoxUniform(low=-2 * torch.ones(num_dims), high=2 * torch.ones(num_dims))
simulator = lambda theta: theta + torch.randn_like(theta) * 0.1

# 2. Generate observed data and calibration data
x_o = torch.tensor([0.5, 0.5, 0.3])  # Test observation
theta_train = prior.sample((num_sims,))
x_train = simulator(theta_train)

# Generate separate calibration data
theta_calib = prior.sample((num_calib,))
x_calib = simulator(theta_calib)

# 3. Train Bayesian model
inference = NPE(prior)
inference.append_simulations(theta_train, x_train).train()

posterior = inference.build_posterior()

# Evaluate log-probabilities for calibration data

log_probs = posterior.log_prob(theta_calib[0, :], x=x_calib[0, :])


# 4. Fit BayCon to HPD score using local conformal
bayes_conf = BayCon(
    sbi_score=HPDScore,
    base_inference=inference,
    is_fitted=True,
    conformal_method="local",
)
bayes_conf.fit(
    X=x_train,
    theta=theta_train,
)

# 5. Computing cutoffs and calibrating
bayes_conf.calib(
    X_calib=x_calib,
    theta_calib=theta_calib,
)

# 7. Get conformal cutoff for test observation
print("\nConformal prediction interval for x_o:")
pred_cutoff = bayes_conf.predict_cutoff(x_o.reshape(1, -1).detach().numpy())
