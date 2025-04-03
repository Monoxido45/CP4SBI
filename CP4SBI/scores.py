from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone
import torch


# defining score basic class
class sbi_Scores(ABC):
    """
    Base class to build any conformity score of choosing.
    In this class, one can define any conformity score for any base model of interest, already fitted or not.
    ----------------------------------------------------------------
    """

    def __init__(self, inference_obj, is_fitted=False, cuda=False):
        self.inference_obj = inference_obj
        self.is_fitted = is_fitted
        self.cuda = cuda

    @abstractmethod
    def fit(self, X, theta):
        """
        Fit the base model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   theta: Training parameter vector.

        Output: Scores object
        """
        pass

    @abstractmethod
    def compute(self, X_calib, theta_calib):
        """
        Compute the conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   theta_calib: Calibration label vector

        Output: Conformity score vector
        """
        pass


# HPD score
class HPDScore(sbi_Scores):
    def fit(self, X=None, thetas=None):
        # setting up model for SBI package
        if not self.is_fitted:
            if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(thetas, torch.Tensor) or thetas.dtype != torch.float32:
                thetas = torch.tensor(thetas, dtype=torch.float32)
            self.inference_obj.append_simulations(thetas, X)
            self.inference_obj.train()

        self.posterior = self.inference_obj.build_posterior()
        return self

    def compute(self, X_calib, thetas_calib):
        if not isinstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if (
            not isinstance(thetas_calib, torch.Tensor)
            or thetas_calib.dtype != torch.float32
        ):
            thetas_calib = torch.tensor(thetas_calib, dtype=torch.float32)

        if self.cuda:
            X_calib = X_calib.to(device="cuda")
            thetas_calib = thetas_calib.to(device="cuda")

        # obtaining posterior estimators
        par_n = thetas_calib.shape[0]
        log_prob_array = np.zeros(par_n)
        for i in range(par_n):
            if not self.cuda:
                log_prob_array[i] = (
                    self.posterior.log_prob(thetas_calib[i, :], x=X_calib[i, :])
                    .detach()
                    .numpy()
                )
            else:
                log_prob_array[i] = (
                    self.posterior.log_prob(thetas_calib[i, :], x=X_calib[i, :])
                    .cpu()
                    .detach()
                    .numpy()
                )

        # computing posterior density for theta
        return -(np.exp(log_prob_array))


# QuantileScore
class QuantileScore(sbi_Scores):
    def fit(self, X=None, thetas=None):
        # setting up model for SBI package
        if not self.is_fitted:
            if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(thetas, torch.Tensor) or thetas.dtype != torch.float32:
                thetas = torch.tensor(thetas, dtype=torch.float32)
            self.inference_obj.append_simulations(thetas, X)
            self.inference_obj.train()

        self.posterior = self.inference_obj.build_posterior()
        return self

    def compute(self, X_calib, thetas_calib, prob=[0.025, 0.095], n_sims=10000):

        if not isinstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if (
            not isinstance(thetas_calib, torch.Tensor)
            or thetas_calib.dtype != torch.float32
        ):
            thetas_calib = torch.tensor(thetas_calib, dtype=torch.float32)

        # computing quantiles for prob
        par_n = thetas_calib.shape[0]
        quantile_array = np.zeros(par_n)
        for i in range(par_n):
            quantiles_samples_theta = np.quantile(
                self.posterior.sample((n_sims,), x=X_calib[i, :]), q=prob, axis=0
            )

            quantile_array[i] = (
                np.max(
                    quantiles_samples_theta[:, 0] - thetas_calib[i, :],
                    thetas_calib[i, :] - quantiles_samples_theta[:, 1],
                )
                .detach()
                .numpy()
            )

        # computing quantile score posterior for theta
        return quantile_array


# TODO: waldo score and quantile score
