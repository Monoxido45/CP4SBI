import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from operator import itemgetter

from tqdm import tqdm


# LOCART class for derivin cutoffs
class LocartInf(BaseEstimator):
    """
    Local Regression Tree.
    Fit LOCART and LOFOREST local calibration methods for any bayesian score and base model of interest. The specification of the score
    can be made through the usage of the basic class "sbi_Scores". Through the "split_calib" parameter we can decide whether to use all calibration set to
    obtain both the partition and cutoffs or split it into two sets, one specific for partitioning and other for obtaining the local cutoffs. Also, if
    desired, we can fit the augmented version of both our method (A-LOCART and A-LOFOREST) by the "weighting" parameter, which if True, adds conditional variance estimates to our feature matrix in the calibration and prediction step.
    ----------------------------------------------------------------
    """

    def __init__(
        self,
        sbi_score,
        base_inference,
        alpha,
        is_fitted=False,
        cart_type="CART",
        split_calib=True,
        weighting=False,
        cuda=False,
    ):
        """
        Input: (i)    sbi_score: Bayesian score of choosing. It can be specified by instantiating a Bayesian score class based on the sbi_Scores basic class.
               (ii)   base_inference: Base SBI inference model to be embedded in the score class.
               (iii)  alpha: Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
               (iv)   base_model_type: Boolean indicating whether the base model ouputs quantiles or not. Default is False.
               (v)    cart_type: Set "CART" to obtain LOCART prediction intervals and "RF" to obtain LOFOREST prediction intervals. Default is CART.
               (vi)   split_calib: Boolean designating if we should split the calibration set into partitioning and cutoff set. Default is True.
               (viii) weighting: Set whether we should augment the feature space with conditional variance estimates. Default is False.
        """
        self.sbi_score = sbi_score(
            base_inference,
            is_fitted=is_fitted,
            cuda=cuda,
        )

        # checking if base model is fitted
        self.base_inference = self.sbi_score.inference_obj
        self.alpha = alpha
        self.cart_type = cart_type
        self.split_calib = split_calib
        self.weighting = weighting
        self.cuda = cuda

    def fit(self, X, theta, random_seed_tree=1250, **kwargs):
        """
        Fit base model embeded in the conformal score class to the training set.
        If "weigthing" is True, we fit a Random Forest model to obtain variance estimations as done in Bostrom et.al.(2021).
        --------------------------------------------------------

        Input: (i)    X: Training numpy feature matrix
               (ii)   y: Training label array
               (iii)  random_seed_tree: Random Forest random seed for variance estimation (if weighting parameter is True).
               (iv)   **kwargs: Keyword arguments passed to fit the random forest used for variance estimation.

        Output: LocartSplit object
        """
        self.sbi_score.fit(X, theta)
        if self.weighting == True:
            # TODO: add better option for weighting and A-locart
            self.dif_model = (
                RandomForestRegressor(random_state=random_seed_tree)
                .set_params(**kwargs)
                .fit(X, theta)
            )
        # TODO: add CDF of the score modification inside LOCART
        return self

    def calib(
        self,
        X_calib,
        theta_calib,
        random_seed=1250,
        prune_tree=True,
        prune_seed=780,
        cart_train_size=0.5,
        **kwargs
    ):
        """
        Calibrate conformity score using CART
        As default, we fix "min_samples_leaf" as 100 for the CART algorithm,meaning that each partition element will have at least
        100 samples each, and use the sklearn default for the remaining parameters. To generate other partitioning schemes, all CART parameters
        can be changed through keyword arguments, but we recommend changing only the "min_samples_leaf" argument if needed.
        --------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix
               (ii)   theta_calib: Calibration parameter array
               (iii)  random_seed: Random seed for CART or Random Forest fitted to the confomity scores.
               (iv)   prune_tree: Boolean indicating whether CART tree should be pruned or not.
               (v)    prune_seed: Random seed set for data splitting in the prune step.
               (vi)   cart_train_size: Proportion of calibration data used in partitioning.
               (vii)    **kwargs: Keyword arguments to be passed to CART or Random Forest.

        Ouput: Vector of cutoffs.
        """
        res = self.sbi_score.compute(X_calib, theta_calib)
        print(np.min(res))
        if self.weighting:
            w = self.compute_difficulty(X_calib)
            X_calib = np.concatenate((X_calib, w.reshape(-1, 1)), axis=1)

        # splitting calibration data into a partitioning set and a cutoff set
        if self.split_calib:
            (
                X_calib_train,
                X_calib_test,
                res_calib_train,
                res_calib_test,
            ) = train_test_split(
                X_calib,
                res,
                test_size=1 - cart_train_size,
                random_state=random_seed,
            )

        if self.cart_type == "CART":
            # declaring decision tree
            self.cart = DecisionTreeRegressor(
                random_state=random_seed, min_samples_leaf=100
            ).set_params(**kwargs)
            # obtaining optimum alpha to prune decision tree
            if prune_tree:
                if self.split_calib:
                    (
                        X_train_prune,
                        X_test_prune,
                        res_train_prune,
                        res_test_prune,
                    ) = train_test_split(
                        X_calib_train,
                        res_calib_train,
                        test_size=0.5,
                        random_state=prune_seed,
                    )
                else:
                    (
                        X_train_prune,
                        X_test_prune,
                        res_train_prune,
                        res_test_prune,
                    ) = train_test_split(
                        X_calib,
                        res,
                        test_size=0.5,
                        random_state=prune_seed,
                    )

                optim_ccp = self.prune_tree(
                    X_train_prune, X_test_prune, res_train_prune, res_test_prune
                )
                # pruning decision tree
                self.cart.set_params(ccp_alpha=optim_ccp)

            # fitting and predicting leaf labels
            if self.split_calib:
                self.cart.fit(X_calib_train, res_calib_train)
                leafs_idx = self.cart.apply(X_calib_test)
            else:
                self.cart.fit(X_calib, res)
                leafs_idx = self.cart.apply(X_calib)

            self.leaf_idx = np.unique(leafs_idx)
            self.cutoffs = {}

            for leaf in self.leaf_idx:
                if self.split_calib:
                    current_res = res_calib_test[leafs_idx == leaf]

                    # correcting 1 - alpha
                    n = current_res.shape[0]

                    self.cutoffs[leaf] = np.quantile(
                        current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )
                else:
                    current_res = res[leafs_idx == leaf]

                    # correcting 1 - alpha
                    n = current_res.shape[0]

                    self.cutoffs[leaf] = np.quantile(
                        current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )

        return self.cutoffs

    def compute_difficulty(self, X):
        """
        Auxiliary function to compute difficulty for each sample.
        --------------------------------------------------------
        input: (i)    X: specified numpy feature matrix

        output: Vector of variance estimates for each sample.
        """
        cart_pred = np.zeros((X.shape[0], len(self.dif_model.estimators_)))
        i = 0
        # computing the difficulty score for each X_score
        for cart in self.dif_model.estimators_:
            cart_pred[:, i] = cart.predict(X)
            i += 1
        # computing variance for each dataset row
        return cart_pred.var(1)

    def prune_tree(self, X_train, X_valid, res_train, res_valid):
        """
        Auxiliary function to conduct decision tree post pruning.
        --------------------------------------------------------
        Input: (i)    X_train: numpy feature matrix used to fit decision trees for each cost complexity alpha values.
               (ii)   X_valid: numpy feature matrix used to validate each cost complexity path.
               (iii)  res_train: conformal scores used to fit decision trees for each cost complexity alpha values.
               (iv)   res_valid: conformal scores used to validate each cost complexity path.

        Output: Optimal cost complexity path to perform pruning.
        """
        prune_path = self.cart.cost_complexity_pruning_path(X_train, res_train)
        ccp_alphas = prune_path.ccp_alphas
        current_loss = float("inf")
        # cross validation by data splitting to choose alphas
        for ccp_alpha in ccp_alphas:
            preds_ccp = (
                clone(self.cart)
                .set_params(ccp_alpha=ccp_alpha)
                .fit(X_train, res_train)
                .predict(X_valid)
            )
            loss_ccp = mean_squared_error(res_valid, preds_ccp)
            if loss_ccp < current_loss:
                current_loss = loss_ccp
                optim_ccp = ccp_alpha

        return optim_ccp

    def plot_locart(self, title=None):
        """
        Plot decision tree feature space partition
        --------------------------------------------------------
        Output: Decision tree plot object.
        """
        if self.cart_type == "CART":
            plot_tree(self.cart, filled=True)
            if title == None:
                plt.title("Decision Tree fitted to non-conformity score")
            else:
                plt.title(title)
            plt.show()

    def predict_cutoff(self, X):
        """
        Predict cutoffs for each test sample using locart local cutoffs.
        --------------------------------------------------------
        Input: (i)    X: test numpy feature matrix

        Output: Cutoffs for each test sample.
        """
        # identifying cutoff point
        if self.weighting:
            w = self.compute_difficulty(X)
            X_tree = np.concatenate((X, w.reshape(-1, 1)), axis=1)
        else:
            X_tree = X

        leaves_idx = self.cart.apply(X_tree)
        cutoffs = np.array(itemgetter(*leaves_idx)(self.cutoffs))

        return cutoffs


class CDFSplit(BaseEstimator):
    """
    CDF split class for conformalizing bayesian credible regions.
    Fit CDF split calibration methods for any bayesian score and base model of interest. The specification of the score
    can be made through the usage of the basic class "sbi_Scores".
    ----------------------------------------------------------------
    """

    def __init__(
        self,
        sbi_score,
        base_inference,
        alpha,
        is_fitted=False,
        cuda=False,
        local_cutoffs=False,
    ):
        """
        Input: (i)    sbi_score: Bayesian score of choosing. It can be specified by instantiating a Bayesian score class based on the sbi_Scores basic class.
               (ii)   base_inference: Base SBI inference model to be embedded in the score class.
               (iii)  alpha: Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
               (iv)   is_fitted: Boolean indicating whether the base model is already fitted or not.
               (v)    cuda: Boolean indicating whether to use GPU or not.
               (vi)   local_cutoffs: Boolean indicating whether to use local cutoffs derived by LOCART or not.
        """
        self.sbi_score = sbi_score(
            base_inference,
            is_fitted=is_fitted,
            cuda=cuda,
        )

        # checking if base model is fitted
        self.base_inference = self.sbi_score.inference_obj
        self.alpha = alpha
        self.cuda = cuda
        self.is_fitted = is_fitted
        self.local_cutoffs = local_cutoffs

    def fit(self, X, theta):
        """
        Fit base model embeded in the conformal score class to the training set.
        --------------------------------------------------------

        Input: (i)    X: Training numpy feature matrix
               (ii)   theta: Training parameter array

        Output: HPDSPlit object
        """
        self.sbi_score.fit(X, theta)
        return self

    def calib(
        self,
        X_calib,
        theta_calib,
        n_samples=1000,
        split_calib=False,
        cart_train_size=0.5,
        random_seed=1250,
        prune_tree=True,
        min_samples_leaf=100,
    ):
        """
        Calibrate conformity score using the cumulative distribution function of the score derived by the sbi base model.

        --------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix
               (ii)   theta_calib: Calibration parameter array
               (iii)  random_seed: Random seed for Monte Carlo.
               (iv)   n_samples: Number of samples to be used for Monte Carlo approximation.
               (v)  split_calib: Boolean indicating whether to split the calibration set into partitioning and cutoff set when using local cutoffs.
               (vi)  cart_train_size: Proportion of calibration data used in partitioning when using local cutoffs.
               (vii) random_seed: Random seed for data splitting in the prune step when using local cutoffs.
               (viii) prune_tree: Boolean indicating whether to prune the decision tree or not when using local cutoffs.

        Ouput: Vector of cutoffs.
        """
        res = self.sbi_score.compute(X_calib, theta_calib)

        # Transform X_calib and theta_calib into tensors if they are numpy arrays
        if isinstance(X_calib, np.ndarray):
            X_calib = torch.tensor(X_calib, dtype=torch.float32)
        if isinstance(theta_calib, np.ndarray):
            theta_calib = torch.tensor(theta_calib, dtype=torch.float32)

        # for each X_calib, we generate n_samples samples from the posterior
        # and compute the new score
        new_res = np.zeros(res.shape[0])
        i = 0
        for X in tqdm(X_calib, desc="Computting new CDF scores"):
            X = X.reshape(1, -1)

            if self.cuda:
                X = X.to(device="cuda")

            # generating n_samples samples from the posterior
            theta_pos = self.sbi_score.posterior.sample(
                (n_samples,),
                x=X,
                show_progress_bars=False,
            )

            # computing the score for each sample
            res_theta = self.sbi_score.compute(X, theta_pos, one_X=True)
            # computing new conformal score
            new_res[i] = np.mean(res_theta <= res[i])

            i += 1

        if not self.local_cutoffs:
            # computing cutoff on new res
            n = new_res.shape[0]
            self.cutoffs = np.quantile(
                new_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
            )
        else:
            print("Fitting LOCART to CDF scores to derive local cutoffs")
            if split_calib:
                (
                    X_calib_train,
                    X_calib_test,
                    res_calib_train,
                    res_calib_test,
                ) = train_test_split(
                    X_calib,
                    new_res,
                    test_size=1 - cart_train_size,
                    random_state=random_seed,
                )

            # fitting LOCART to obtain local cutoffs
            # instatiating decision tree
            self.cart = DecisionTreeRegressor(
                random_state=random_seed, min_samples_leaf=min_samples_leaf
            )

            # obtaining optimum alpha to prune decision tree used to obtain local cutoffs
            if split_calib and prune_tree:
                optim_ccp = self.prune_tree(
                    X_calib_train,
                    res_calib_train,
                    split_size=0.5,
                    random_seed=random_seed,
                )
                # pruning decision tree
                self.cart.set_params(ccp_alpha=optim_ccp)
            elif prune_tree:
                optim_ccp = self.prune_tree(
                    X_calib,
                    new_res,
                    split_size=0.5,
                    random_seed=random_seed,
                )
                # pruning decision tree
                self.cart.set_params(ccp_alpha=optim_ccp)

            # fitting and predicting leaf labels
            if split_calib:
                self.cart.fit(X_calib_train, res_calib_train)
                leafs_idx = self.cart.apply(X_calib_test)
            else:
                self.cart.fit(X_calib, new_res)
                leafs_idx = self.cart.apply(X_calib)

            self.leaf_idx = np.unique(leafs_idx)
            self.cutoffs = {}

            for leaf in self.leaf_idx:
                if split_calib:
                    current_res = res_calib_test[leafs_idx == leaf]

                    # correcting 1 - alpha
                    n = current_res.shape[0]

                    self.cutoffs[leaf] = np.quantile(
                        current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )
                else:
                    current_res = new_res[leafs_idx == leaf]

                    # correcting 1 - alpha
                    n = current_res.shape[0]

                    self.cutoffs[leaf] = np.quantile(
                        current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                    )
        return self.cutoffs

    # function to prune tree
    def prune_tree(self, X, res, split_size, random_seed):
        """
        Auxiliary function to conduct decision tree post pruning.
        --------------------------------------------------------
        Input: (i)    X: numpy feature matrix used to fit decision trees for each cost complexity alpha values.
               (ii)  res: conformal scores used to fit decision trees for each cost complexity alpha values.
               (iii) split_size: proportion of data used to fit decision trees for each cost complexity alpha values.
                (iv)  random_seed: random seed for data splitting in the prune step.

        Output: Optimal cost complexity path to perform pruning.
        """
        # splitting data into training and validation sets
        (
            X_train,
            X_valid,
            res_train,
            res_valid,
        ) = train_test_split(
            X,
            res,
            test_size=split_size,
            random_state=random_seed,
        )

        prune_path = self.cart.cost_complexity_pruning_path(X_train, res_train)
        ccp_alphas = prune_path.ccp_alphas
        current_loss = float("inf")
        # cross validation by data splitting to choose alphas
        for ccp_alpha in ccp_alphas:
            preds_ccp = (
                clone(self.cart)
                .set_params(ccp_alpha=ccp_alpha)
                .fit(X_train, res_train)
                .predict(X_valid)
            )
            loss_ccp = mean_squared_error(res_valid, preds_ccp)
            if loss_ccp < current_loss:
                current_loss = loss_ccp
                optim_ccp = ccp_alpha

        return optim_ccp

    def predict_cutoff(self, X_test, n_samples=2000):
        """
        Predict cutoffs for each test sample using the CDF conformal method
        --------------------------------------------------------
        Input: (i)    X: test numpy feature matrix

        Output: Cutoffs for each test sample.
        """
        cutoffs = np.zeros(X_test.shape[0])
        i = 0

        # Transform X_test into a tensor if it is a numpy array
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        # sampling from posterior
        if not self.local_cutoffs:
            for X in tqdm(X_test, desc="Computting CDF-based cutoffs"):
                X = X.reshape(1, -1)
                if self.cuda:
                    X = X.to(device="cuda")
                theta_pos = self.sbi_score.posterior.sample(
                    (n_samples,),
                    x=X,
                    show_progress_bars=False,
                )

                # computing the quantile from theta_pos using the new cutoff
                cutoffs[i] = np.quantile(
                    self.sbi_score.compute(X, theta_pos, one_X=True),
                    q=self.cutoffs,
                )

                i += 1
        else:
            # first deriving the local cutoffs
            leaves_idx = self.cart.apply(X_test.numpy())
            print(leaves_idx)
            cutoffs_local = np.array(itemgetter(*leaves_idx)(self.cutoffs))

            for X in tqdm(X_test, desc="Computting local CDF-based cutoffs"):
                X = X.reshape(1, -1)

                if cutoffs_local.ndim == 0:
                    spec_cutoff = cutoffs_local
                else:
                    spec_cutoff = cutoffs_local[i]

                if self.cuda:
                    X = X.to(device="cuda")

                theta_pos = self.sbi_score.posterior.sample(
                    (n_samples,),
                    x=X,
                    show_progress_bars=False,
                )

                # computing the quantile from theta_pos using the new cutoff
                cutoffs[i] = np.quantile(
                    self.sbi_score.compute(X, theta_pos, one_X=True),
                    q=spec_cutoff,
                )

                i += 1

        return cutoffs


# class for conformalizing bayesian credible regions
class BayCon:
    def __init__(
        self,
        sbi_score,
        base_inference,
        is_fitted=False,
        conformal_method="global",
        alpha=0.1,
        split_calib=False,
        weighting=False,
        cuda=False,
    ):
        """
        Class for computing statistical scores.

        Args:
            sbi_score: Bayesian score class instance
            base_inference: Base inference model to be used
            is_fitted (bool): Flag indicating if the model is fitted
            conformal_method (str): Method for conformal prediction ('global', 'local', "CDF" or "CDF local")
        """
        self.is_fitted = is_fitted
        self.cuda = cuda
        self.conformal_method = conformal_method
        self.sbi_score = sbi_score(
            base_inference,
            is_fitted=is_fitted,
            cuda=self.cuda,
        )
        self.base_inference = base_inference
        # checking if base model is fitted
        self.alpha = alpha

        if self.conformal_method == "local":
            self.locart = LocartInf(
                sbi_score,
                base_inference,
                alpha=self.alpha,
                is_fitted=self.is_fitted,
                split_calib=split_calib,
                weighting=weighting,
                cuda=cuda,
            )
        elif self.conformal_method == "CDF":
            self.cdf_split = CDFSplit(
                sbi_score,
                base_inference,
                alpha=self.alpha,
                is_fitted=self.is_fitted,
                cuda=cuda,
            )
        elif self.conformal_method == "CDF local":
            self.cdf_split = CDFSplit(
                sbi_score,
                base_inference,
                alpha=self.alpha,
                is_fitted=self.is_fitted,
                cuda=cuda,
                local_cutoffs=True,
            )

    def fit(self, X, theta):
        """
        Fit the SBI score to the training data.

        Args:
            X: Training feature matrix
            theta: Training parameter vector
        """
        if self.conformal_method == "local":
            self.locart.fit(X, theta)
        elif self.conformal_method == "CDF" or self.conformal_method == "CDF local":
            self.cdf_split.fit(X, theta)
        else:
            self.sbi_score.fit(X, theta)
        return self

    def calib(
        self,
        X_calib,
        theta_calib,
        split_calib=False,
        prune_tree=True,
        min_samples_leaf=100,
        cart_train_size=0.5,
        random_seed=1250,
        locart_kwargs=None,
    ):
        """
        Calibrate the credible region using the calibration set.

        Args:
            X_calib: Calibration feature matrix
            theta_calib: Calibration parameter vector
            locart_kwargs: Additional arguments for LOCART calibration. Must be in a dictionary format with each entry being a parameter of interest.

        Raises:
            RuntimeError: If called with empty data
        """
        # Ensure X_calib and theta_calib are numpy arrays
        if isinstance(X_calib, torch.Tensor):
            X_calib = X_calib.numpy()
        if isinstance(theta_calib, torch.Tensor):
            theta_calib = theta_calib.numpy()

        if len(X_calib) == 0 or len(theta_calib) == 0:
            raise RuntimeError("Calibration data cannot be empty")

        # computing cutoffs using standard approach
        if self.conformal_method == "global":
            res = self.sbi_score.compute(X_calib, theta_calib)
            n = res.shape[0]

            # computing cutoff
            self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)

        # computing cutoffs using LOCART
        elif self.conformal_method == "local":
            self.locart.fit(X_calib, theta_calib)
            if locart_kwargs is not None:
                self.cutoff = self.locart.calib(
                    X_calib,
                    theta_calib,
                    prune_tree=prune_tree,
                    split_calib=split_calib,
                    min_samples_leaf=min_samples_leaf,
                    cart_train_size=cart_train_size,
                    random_seed=random_seed,
                    **locart_kwargs,
                )
            else:
                self.cutoff = self.locart.calib(
                    X_calib,
                    theta_calib,
                    prune_tree=prune_tree,
                    split_calib=split_calib,
                    min_samples_leaf=min_samples_leaf,
                    cart_train_size=cart_train_size,
                    random_seed=random_seed,
                )

        elif self.conformal_method == "CDF":
            self.cutoff = self.cdf_split.calib(X_calib, theta_calib)

        elif self.conformal_method == "CDF local":
            self.cutoff = self.cdf_split.calib(
                X_calib,
                theta_calib,
                prune_tree=prune_tree,
                split_calib=split_calib,
                min_samples_leaf=min_samples_leaf,
                cart_train_size=cart_train_size,
                random_seed=random_seed,
            )

        return self.cutoff

    def predict_cutoff(
        self,
        X_test,
    ):
        """self.is_fitted = True
        Predict cutoffs for test samples using the calibrated conformal method.
        Args:
        X_test (numpy.ndarray): Test feature matrix.
        numpy.ndarray: Predicted cutoffs for each test sample.

        RuntimeError: If the conformal method is not calibrated before calling this function.
        """
        if self.conformal_method == "local":
            if self.locart is None:
                raise RuntimeError(
                    "Conformal method must be calibrated before prediction"
                )
            cutoffs = self.locart.predict_cutoff(X_test)
        elif self.conformal_method == "global":
            cutoffs = np.repeat(self.cutoff, X_test.shape[0])

        elif self.conformal_method == "CDF" or self.conformal_method == "CDF local":
            cutoffs = self.cdf_split.predict_cutoff(X_test)

        return cutoffs
