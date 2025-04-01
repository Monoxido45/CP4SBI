import CP4SBI
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from CP4SBI.baycon import BayesianInference
from CP4SBI.baycon import BayCon


# Example usage:
num_dims = 3
num_sims = 1000
num_calib = 500  # Number of calibration points

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
BI_model = BayesianInference(NPE, prior)
BI_model.fit(x_train)

# 4. Sample from posterior for test observation
print("Posterior samples for x_o:")
print(BI_model.sample_posterior(x_o, num_sims=5))  # Just show 5 samples

# 5. Create score model and compute score
Score_model = BayCon(BI_model, score_type="CQR")  # Using CQR score
test_theta = theta_train[0]  # Using first theta for demonstration
print("\nScore for test point:")
print(Score_model.compute_score(x_o, test_theta))

# 6. Calibrate using calibration set
print("\nCalibrating with calibration set...")
Score_model.calib(x_calib, theta_calib, alpha=0.5)  # 95% coverage
print(f"Conformal threshold (t): {Score_model.t_conformal}")

# 7. # Get conformal prediction interval for test observation
print("\nConformal prediction interval for x_o:")
pred_interval = Score_model.interval_conformal(x_o)

# Print formatted results and verify if true parameter is in the interval
for dim in range(num_dims):
    print(f"\nDimension {dim+1}:")
    print(
        f"  Interval: [{pred_interval['interval'][dim,0]:.3f}, {pred_interval['interval'][dim,1]:.3f}]"
    )
    print(f"  Width: {pred_interval['width'][dim]:.3f}")
    print(f"  True theta used for x_o simulation: {test_theta.numpy()[dim]}")
    print(
        f"  Contains true theta: {pred_interval['interval'][dim,0] <= test_theta[dim] <= pred_interval['interval'][dim,1]}"
    )

print(f"\nNominal coverage level: {pred_interval['coverage_level']}")

# 8. Verify if true parameter is in the interval
# print(f"\nTrue theta used for x_o simulation: {test_theta.numpy()}")
# print("Is true theta in interval for each dimension:")
# for dim in range(num_dims):
#    in_interval = pred_interval['interval'][dim, 0] <= test_theta[dim].item() <= pred_interval['interval'][dim, 1]
#    print(f"  Dimension {dim+1}: {in_interval}")
