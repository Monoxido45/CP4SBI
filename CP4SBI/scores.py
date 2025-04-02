from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone
from tqdm import tqdm


# defining score basic class
class Scores(ABC):
    """
    Base class to build any conformity score of choosing.
    In this class, one can define any conformity score for any base model of interest, already fitted or not.
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, is_fitted, **kwargs):
        self.is_fitted = is_fitted
        if self.is_fitted:
            self.base_model = base_model
        elif base_model is not None:
            self.base_model = base_model(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the base model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   y: Training label vector.

        Output: Scores object
        """
        pass

    @abstractmethod
    def compute(self, X_calib, y_calib):
        """
        Compute the conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   y_calib: Calibration label vector

        Output: Conformity score vector
        """
        pass

    @abstractmethod
    def predict(self, X_test, cutoff):
        """
        Compute prediction intervals specified cutoff(s).
        --------------------------------------------------------
        Input: (i)    X_test: Test feature matrix
               (ii)   cutoff: Cutoff vector

        Output: Prediction intervals for test sample.
        """
        pass


# basic lambda score that does not need to be estimated
class LambdaScore(Scores):
    def fit(self, X, y):
        return self

    def compute(self, thetas, lambdas):
        return lambdas

    def predict(self, thetas, cutoff):
        pred = np.vstack((thetas - cutoff, thetas + cutoff)).T
        return pred
    
    
# QuantileScore
class QuantileScore(Scores):
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
    
    def compute(self, X_calib, thetas_calib, prob = [0.025, 0.095], n_sims = 10000):
        
        if not isnstance(X_calib, torch.Tensor) or X_calib.dtype != torch.float32:
            X_calib = torch.tensor(X_calib, dtype = torch.float32)
        if not isnstance(thetas_calib, torch.Tensor) or thetas_calib.dtype != torch.float32:
            thetas_calib = torch.tensor(thetas_calib, dtype = torch.float32)
            
        # computing quantiles for prob
        par_n = thetas_calib.shape[0]
        quantile_array = np.zeros(par_n)
        for i in range(par_n):
            quantiles_samples_theta = np.quantile(self.posterior.sample((n_sims,), x=X_calib[i, :]), q = prob, axis = 0)
            
            quantile_array[i] = (np.max(quantiles_samples_theta[:,0] - thetas_calib[i, :],
                                        thetas_calib[i, :] - quantiles_samples_theta[:,1])
                .detach()
                .numpy()
            )
        
        # computing quantile score posterior for theta
        return quantile_array
        
        
    
