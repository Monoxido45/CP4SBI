import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns


# class for performing conformal calibration


class BayesianInference:
    """
    A class for performing Bayesian inference using specified methods and priors.

    Attributes:
        method: Bayesian inference method to be used
        prior: Prior distribution for the model
        posterior: Posterior distribution after training
        is_trained (bool): Flag indicating whether the model has been trained
    """

    def __init__(self, method, prior):
        """
        Initialize the Bayesian inference model.

        Args:
            method: Inference method to use
            prior: Prior distribution for the parameters
        """
        self.method = method
        self.prior = prior
        self.is_trained = False

    def fit(self, x, num_sims=1000):
        """
        Train the model by fitting the posterior distribution.

        Args:
            x: Observed data
            num_sims (int): Number of simulations to run
        """
        inference = self.method(self.prior)
        theta = self.prior.sample((num_sims,))
        inference.append_simulations(theta, x).train()

        self.posterior = inference.build_posterior()
        self.is_trained = True

    def sample_posterior(self, x, num_sims=1000):
        """
        Sample from the trained posterior distribution.

        Args:
            x: Data to condition the sampling (single observation)
            num_sims (int): Number of samples to generate

        Returns:
            Samples from the posterior distribution as numpy array

        Raises:
            RuntimeError: If called before training the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before sampling")

        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        samples = self.posterior.sample((num_sims,), x=x)

        return samples.numpy()

    def quantile_posterior(self, x, alpha, num_sims=1000):
        """
        Args:
            x: data to condition the sampling
            alpha: vector with alphas values for compute quantile
            num_sims (int): Number of samples to generate

        Returns:
            Quantiles alpha for each value of x using MC quantile of sample_posterior
        """
        if any((a < 0) or (a > 1) for a in alpha):
            raise ValueError("All alpha values must be between 0 and 1")

        # Get posterior samples
        samples = self.sample_posterior(x, num_sims)

        # Compute quantiles
        quantiles = np.quantile(samples, alpha, axis=0)

        return quantiles

    def mean_posterior(self, x, num_sims=1000):
        """
        Args:
            x: data to condition the sampling
            num_sims (int): Number of samples to generate

        Returns:
            Mean for each value of x using MC quantile of sample_posterior
        """
        # Get posterior samples
        samples = self.sample_posterior(x, num_sims)

        # Compute mean
        mean = np.mean(samples, axis=0)

        return mean


class BayCon:
    def __init__(self, model, score_type="CQR"):
        """
        Class for computing statistical scores.

        Args:
            model: Bayesian inference model instance
            score_type (str): Type of score to compute ('CQR' or 'RS')
        """
        self.BI_model = model
        self.score_type = score_type

    def compute_score(self, x, theta, prob=[0.025, 0.975]):
        """
        Computes the score based on the specified type.

        Args:
            x: Input data
            theta: Parameter value
            prob (list): Significance levels (for CQR score)

        Returns:
            Computed score

        Raises:
            ValueError: If unknown score type is specified
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(theta, torch.Tensor):
            theta = theta.numpy()

        if self.score_type == "CQR":
            quantile_score = self.BI_model.quantile_posterior(x, prob)
            return np.maximum(quantile_score[0] - theta, theta - quantile_score[1])
        elif self.score_type == "RS":
            mean_score = self.BI_model.mean_posterior(x)
            return np.abs(theta - mean_score)
        else:
            raise ValueError("Unknown score type. Choose either 'CQR' or 'RS'")

    def calib(self, x, theta, alpha=0.05):
        """
        Compute the conformal threshold for prediction intervals.

        Args:
            x: Input data (calibration set)
            theta: True parameters (calibration set)
            alpha: Significance level (1 - coverage)

        Raises:
            RuntimeError: If called with empty data
        """
        if len(x) == 0 or len(theta) == 0:
            raise RuntimeError("Calibration data cannot be empty")

        # Compute scores for all calibration points
        scores = np.array([self.compute_score(x[i], theta[i]) for i in range(len(x))])

        # Calculate the (1-alpha) quantile of scores
        self.t_conformal = np.quantile(scores, 1 - alpha, method="higher")

    def interval_conformal(self, x, alpha=0.05):
        """
        Compute conformal prediction interval for new observation.

        Args:
            x: New observation
            alpha: Significance level (1 - coverage)

        Returns:
            Dictionary containing:
            - 'interval': List of [lower, upper] bounds for each dimension
            - 'contains_true': Boolean array indicating if test_theta is within bounds
            - 'width': Array of interval widths for each dimension
        """
        if not hasattr(self, "t_conformal"):
            raise RuntimeError("Must run calib() first to compute t_conformal")

        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if self.score_type == "CQR":
            quantile_score = self.BI_model.quantile_posterior(
                x, [alpha / 2, 1 - alpha / 2]
            )
            lower = quantile_score[0] - self.t_conformal
            upper = quantile_score[1] + self.t_conformal
        elif self.score_type == "RS":
            mean_score = self.BI_model.mean_posterior(x)
            lower = mean_score - self.t_conformal
            upper = mean_score + self.t_conformal
        else:
            raise ValueError("Unknown score type. Choose either 'CQR' or 'RS'")

        # Convert to more readable output format
        return {
            "interval": np.stack([lower, upper], axis=1),
            "width": upper - lower,
            "coverage_level": 1 - alpha,
        }
