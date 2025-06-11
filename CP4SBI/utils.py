import numpy as np
import torch
from tqdm import tqdm
from CP4SBI.scores import HPDScore, KDE_HPDScore
from scipy.stats import gaussian_kde


# defining naive function
def naive_method(
    post_estim,
    X,
    alpha=0.1,
    score_type="HPD",
    B_naive=1000,
    device="cuda",
    B_waldo=1000,
    grid_step=0.005,
    n_grid=None,
    kde=False,
):
    """
    Naive credible sets based on the posterior distribution.

    Args:
        data (np.ndarray or torch.Tensor): Input data for the method.
        params (dict): Parameters required for the method.

    Returns:
        result: The result of the naive computation.
    """
    if device == "cuda":
        X = X.to(device="cuda")

    # check if X has only one dimension
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    # samples to compute scores
    samples = post_estim.sample(
        (B_naive,),
        x=X,
        show_progress_bars=False,
    )

    # if score_type is HPD
    if score_type == "HPD":
        if kde:
            # using kde to compute the density
            kde_obj = gaussian_kde(samples.cpu().numpy().T, bw_method="scott")
            conf_scores = -kde_obj.evaluate(samples.cpu().numpy().T)
        else:
            conf_scores = -np.exp(
                post_estim.log_prob(
                    samples,
                    x=X,
                )
                .cpu()
                .numpy()
            )
    elif score_type == "WALDO":
        conf_scores = np.zeros(B_naive)

        # sampling from posterior to compute mean and covariance matrix
        sample_generated = (
            post_estim.sample(
                (B_waldo,),
                x=X,
                show_progress_bars=False,
            )
            .cpu()
            .detach()
            .numpy()
        )

        # computing mean and covariance matrix
        mean_array = np.mean(sample_generated, axis=0)
        covariance_matrix = np.cov(sample_generated, rowvar=False)
        if mean_array.shape[0] > 1:
            inv_matrix = np.linalg.inv(covariance_matrix)
        else:
            inv_matrix = 1 / covariance_matrix

        # computing waldo scores for each estimated posterior sample
        for i in range(B_naive):
            if mean_array.shape[0] > 1:
                sample_fixed = samples[i, :].cpu().numpy()
                conf_scores[i] = (
                    (mean_array - sample_fixed).transpose()
                    @ inv_matrix
                    @ (mean_array - sample_fixed)
                )
            else:
                sample_fixed = samples[i].cpu().numpy()
                conf_scores[i] = (mean_array - samples[i]) ** 2 / (covariance_matrix)

    # picking large grid between maximum and minimum densities
    if n_grid is None:
        t_grid = np.arange(
            np.min(conf_scores),
            np.max(conf_scores),
            grid_step,
        )
    else:
        t_grid = np.linspace(
            np.min(conf_scores),
            np.max(conf_scores),
            num=n_grid,
        )
    target_coverage = 1 - alpha

    # computing MC integral for all t_grid
    coverage_array = np.zeros(t_grid.shape[0])
    for t in t_grid:
        coverage_array[t_grid == t] = np.mean(conf_scores <= t)

    closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
    # finally, finding the naive cutoff
    closest_t = t_grid[closest_t_index]

    if score_type == "WALDO":
        return closest_t, mean_array, inv_matrix
    else:
        return closest_t


# adapting code for HDR
# code extracted and adapted from https://github.com/YoungseogChung/multi-dimensional-recalibration/tree/main
def hdr_alpha(query_val, base_pdf_values):
    assert len(base_pdf_values.shape) == 1
    rank_among_pdfs = np.mean(base_pdf_values >= query_val)

    return rank_among_pdfs


class conditional_hdr_recalibration:
    def __init__(
        self,
        dim_y,
        fhyh,  # (num_data, num_samples, 1) or (num_data, num_samples)
        fhys,  # (num_data, 1) or (num_data, )
        hdr_delta=0.05,
    ):
        self.dim_y = int(dim_y)
        self.hdr_delta = hdr_delta
        num_data, num_samples = fhyh.shape[:2]
        assert fhys.shape[0] == num_data

        self.fhyh = fhyh.reshape(num_data, num_samples)
        self.fhys = fhys.reshape(num_data, 1)

        self.hdr_levels = np.sort(np.mean(self.fhyh >= self.fhys, axis=1))
        self.hdr_level_bounds = np.arange(0, 1 + hdr_delta, hdr_delta)
        self.num_bounds = len(self.hdr_level_bounds) - 1
        self.num_in_bucket = np.histogram(self.hdr_levels, bins=self.hdr_level_bounds)[
            0
        ]
        self.prop_in_bucket = self.num_in_bucket / np.sum(self.num_in_bucket)
        self.sample_scale_constant = 1 / np.max(self.prop_in_bucket)
        self.sample_prop_per_bucket = self.sample_scale_constant * self.prop_in_bucket

        self.min_req_samples = int(
            (self.num_bounds - 1) / np.max(self.sample_prop_per_bucket)
        )

    @property
    def mace(self):
        exp_props = np.linspace(0, 1, len(self.hdr_levels))
        obs_props = np.sort(np.array(self.hdr_levels).flatten())
        mace = np.mean(np.abs(exp_props - obs_props))
        return mace

    def check_torch_attributes(
        self,
        device,
    ):
        if not hasattr(self, "hdr_level_bounds_torch"):
            self.hdr_level_bounds_torch = torch.from_numpy(self.hdr_level_bounds).to(
                device
            )
        if not hasattr(self, "sample_prop_per_bucket_torch"):
            self.sample_prop_per_bucket_torch = torch.from_numpy(
                self.sample_prop_per_bucket
            ).to(device)
        # if hasattr(self, 'mean_bias') and not hasattr(self, 'mean_bias_tensor'):
        #     self.mean_bias_tensor = torch.from_numpy(self.mean_bias).to(device)
        # if hasattr(self, 'error_corr_mat') and not hasattr(self, 'error_corr_mat_tensor'):
        #     self.error_corr_mat_tensor = torch.from_numpy(self.error_corr_mat).to(device)

    def get_rejection_prob(
        self,
        y_hat,  # (num_data, num_samples, dim_y)
        f_hat_y_hat,  # (num_data, num_samples)
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)
        num_samples_per_bucket_arr = (
            hdr_level_bound_idxs[1:] - hdr_level_bound_idxs[:-1]
        )

        order_fhyh = np.flip(
            np.argsort(f_hat_y_hat, axis=1), axis=1
        )  # (num_data, num_samples)

        pt_idxs_per_hdr = []
        for bin_idx in range(self.num_bounds):
            pt_idxs_per_hdr.append(
                order_fhyh[
                    :, hdr_level_bound_idxs[bin_idx] : hdr_level_bound_idxs[bin_idx + 1]
                ]
                # (num_data, c_bin)
            )

        fhyh_sample_prob = -1 * np.ones_like(f_hat_y_hat)
        for bin_idx in range(self.num_bounds):
            for data_idx in range(num_data):
                curr_chunk = fhyh_sample_prob[
                    data_idx, pt_idxs_per_hdr[bin_idx][data_idx]
                ]
                np.testing.assert_equal(curr_chunk, -1)
                fhyh_sample_prob[data_idx, pt_idxs_per_hdr[bin_idx][data_idx]] = (
                    self.sample_prop_per_bucket[bin_idx]
                )
        assert np.all(fhyh_sample_prob.flatten() > 0)

        return fhyh_sample_prob

    def recal_sample(
        self,
        y_hat,  # (num_data, num_samples, dim_y)
        f_hat_y_hat,  # (num_data, num_samples)
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)
        num_samples_per_bucket_arr = (
            hdr_level_bound_idxs[1:] - hdr_level_bound_idxs[:-1]
        )

        order_fhyh = np.flip(
            np.argsort(f_hat_y_hat, axis=1), axis=1
        )  # (num_data, num_samples)

        pt_idxs_per_hdr = []
        for bin_idx in range(self.num_bounds):
            pt_idxs_per_hdr.append(
                order_fhyh[
                    :, hdr_level_bound_idxs[bin_idx] : hdr_level_bound_idxs[bin_idx + 1]
                ]  # (num_data, c_bin)
            )

        num_collect_per_bucket_arr = (
            self.sample_prop_per_bucket * num_samples_per_bucket_arr
        )
        # array times array

        rand_idx_per_bucket = [
            np.random.choice(
                num_samples_per_bucket_arr[bin_idx],
                size=int(num_collect_per_bucket_arr[bin_idx]),
                replace=False,
            )
            # the following line is only for testing
            # #np.arange(int(num_collect_per_bucket_arr[bin_idx]))
            for bin_idx in range(self.num_bounds)  # TODO: this is probably a class att
        ]

        recal_sample_idxs = [
            pt_idxs_per_hdr[bin_idx][:, rand_idx_per_bucket[bin_idx]]
            for bin_idx in range(self.num_bounds)  # TODO: this is probably a class att
        ]  # each element is (num_data, randselected_bin)

        recal_sample_idxs = np.concatenate(
            recal_sample_idxs, axis=1
        )  # (num_data, sum of randselected_bin)

        # import pdb; pdb.set_trace() #TODO: have a bit of a problem here with array slicing
        # y_hat is (num_data, num_samples, dim_y)
        recal_samples = np.stack(
            [
                y_hat[data_idx][recal_sample_idxs[data_idx]]
                for data_idx in range(num_data)
            ]
        )
        recal_samples_f_hat_y_hat = np.stack(
            [
                f_hat_y_hat[data_idx][recal_sample_idxs[data_idx]]
                for data_idx in range(num_data)
            ]
        )

        return recal_samples, recal_samples_f_hat_y_hat

    def torch_recal_sample(
        self,
        y_hat: torch.Tensor,  # (num_data, num_samples, dim_y)
        f_hat_y_hat: torch.Tensor,  # (num_data, num_samples),
        device="cpu",
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (
            self.hdr_level_bounds_torch * num_samples
        ).int()  # tensor
        num_samples_per_bucket_arr = (
            hdr_level_bound_idxs[1:] - hdr_level_bound_idxs[:-1]
        )  # tensor

        order_fhyh = torch.flip(
            torch.argsort(f_hat_y_hat, dim=1), dims=(1,)
        )  # (num_data, num_samples), tensor

        pt_idxs_per_hdr = []  # list of tensors
        for bin_idx in range(self.num_bounds):
            pt_idxs_per_hdr.append(
                order_fhyh[
                    :, hdr_level_bound_idxs[bin_idx] : hdr_level_bound_idxs[bin_idx + 1]
                ]  # (num_data, c_bin)
            )

        num_collect_per_bucket_arr = (
            self.sample_prop_per_bucket_torch * num_samples_per_bucket_arr
        )  # tensor
        # array times array

        rand_idx_per_bucket = [
            torch.randint(
                low=0,
                high=num_samples_per_bucket_arr[bin_idx],
                size=(int(num_collect_per_bucket_arr[bin_idx]),),
            )
            # the following line is only for testing
            # #torch.arange(int(num_collect_per_bucket_arr[bin_idx]))
            for bin_idx in range(self.num_bounds)
        ]  # list of tensors

        recal_sample_idxs = [
            pt_idxs_per_hdr[bin_idx][:, rand_idx_per_bucket[bin_idx]]
            for bin_idx in range(self.num_bounds)
        ]  # each element is (num_data, randselected_bin)

        recal_sample_idxs = torch.cat(
            recal_sample_idxs, dim=1
        )  # (num_data, sum of randselected_bin)

        # import pdb; pdb.set_trace() #TODO: have a bit of a problem here with array slicing
        # y_hat is (num_data, num_samples, dim_y)
        recal_samples = torch.stack(
            [
                y_hat[data_idx][recal_sample_idxs[data_idx]]
                for data_idx in range(num_data)
            ],
            dim=0,
        )
        recal_samples_f_hat_y_hat = torch.stack(
            [
                f_hat_y_hat[data_idx][recal_sample_idxs[data_idx]]
                for data_idx in range(num_data)
            ],
            dim=0,
        )

        return recal_samples, recal_samples_f_hat_y_hat

    def produce_recal_samples_from_mean_std(
        self,
        means,
        stds,
        mean_bias,
        std_ratio,
        error_corr_mat,
        num_samples,
        device,
    ):
        """
        Produce recalibrated samples
        Assumes TORCH!
        Args:
            dim_y:
            means: (num_pts, dim_y)
            stds: (num_pts, dim_y)
            mean_bias: (dim_y,)
            std_ratio:
            error_corr_mat: (dim_y, dim_y)
            num_samples:
            hdr_recal:
            device:

        Returns:

        """
        raise RuntimeError("do not use this function")
        num_pts = means.shape[0]
        # print(means.shape, stds.shape, mean_bias.shape, error_corr_mat.shape)
        check_shape(means, (num_pts, self.dim_y), raise_error=True)
        check_shape(stds, (num_pts, self.dim_y), raise_error=True)
        assert check_shape(mean_bias, (self.dim_y,), raise_error=False) or check_shape(
            mean_bias, (self.dim_y, 1), raise_error=False
        )
        check_shape(error_corr_mat, (self.dim_y, self.dim_y), raise_error=True)

        # checking for torch attributes
        self.check_torch_attributes(device)

        bias_adj_mean = means - mean_bias
        ratio_adj_std = std_ratio * stds
        corr_adj_cov = (
            ratio_adj_std[:, np.newaxis, :]
            * error_corr_mat
            * ratio_adj_std[:, :, np.newaxis]
        )
        rv = make_batch_multivariate_normal_torch(bias_adj_mean, corr_adj_cov, device)

        yh = rv.sample((num_samples,))  # (num_samples, num_pts, dim_y)
        fhyh = rv.log_prob(yh).exp()  # (num_samples, num_pts)

        yh = yh.swapaxes(0, 1)  # (num_pts, num_samples)
        fhyh = fhyh.T  # (num_pts, num_samples)

        recal_yh, recal_fhyh = self.torch_recal_sample(yh, fhyh)

        return recal_yh, recal_fhyh

    def orig_recal_sample(  ### DEPRECATED ###
        self,
        y_hat,  # (num_data, num_samples, dim_y)
        f_hat_y_hat,  # (num_data, num_samples)
    ):
        num_data = y_hat.shape[0]
        num_samples = y_hat.shape[1]
        assert f_hat_y_hat.shape == (num_data, num_samples)

        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)

        recal_samples_per_x = []
        recal_samples_f_hat_y_hat_per_x = []
        for x_idx in tqdm.tqdm(range(num_data)):
            order_fhyh = np.argsort(f_hat_y_hat[x_idx])[::-1]
            pt_idxs_per_hdr = []
            for bin_idx in range(self.num_bounds):
                pt_idxs_per_hdr.append(
                    order_fhyh[
                        hdr_level_bound_idxs[bin_idx] : hdr_level_bound_idxs[
                            bin_idx + 1
                        ]
                    ]
                )

            recal_sample_idxs = []
            for bin_idx, prop in enumerate(self.sample_prop_per_bucket):
                num_samples_in_bucket = pt_idxs_per_hdr[bin_idx].shape[0]
                curr_num_collect = int(prop * num_samples_in_bucket)
                rand_idx = np.random.choice(
                    num_samples_in_bucket, size=curr_num_collect, replace=False
                )
                # rand_idx = np.arange(curr_num_collect)
                recal_sample_idxs.append(pt_idxs_per_hdr[bin_idx][rand_idx])

            recal_sample_idxs = np.concatenate(recal_sample_idxs)
            recal_samples = y_hat[x_idx][recal_sample_idxs]
            recal_samples_f_hat_y_hat = f_hat_y_hat[x_idx][recal_sample_idxs]

            recal_samples_per_x.append(recal_samples)
            recal_samples_f_hat_y_hat_per_x.append(recal_samples_f_hat_y_hat)

        # at each x, number of recal samples are the same
        recal_samples_per_x = np.stack(recal_samples_per_x)
        recal_samples_f_hat_y_hat_per_x = np.stack(recal_samples_f_hat_y_hat_per_x)

        return recal_samples_per_x, recal_samples_f_hat_y_hat_per_x

    def rejection_sample(
        self,
        f_val_input_point,  # scalar
        f_hat_y_hat,  # (N,)
    ):
        num_samples = f_hat_y_hat.shape[0]
        hdr_level_bound_idxs = (self.hdr_level_bounds * num_samples).astype(int)
        order_fhyh = np.argsort(f_hat_y_hat)[::-1]

        input_point_percentile = np.mean(f_hat_y_hat >= f_val_input_point)

        bin_idx = np.digitize(
            input_point_percentile, self.hdr_level_bounds
        )  # gets end index of bin
        bin_sample_prop = self.sample_prop_per_bucket[bin_idx - 1]
        return np.random.uniform() <= bin_sample_prop


# Method for deriving credible regions using HDR recalibrated
def hdr_method(
    post_estim,
    X_calib,
    thetas_calib,
    X_test,
    X_train=None,
    theta_train=None,
    alpha=0.1,
    score_type="HPD",
    B_recal=1000,
    device="cuda",
    grid_step=0.005,
    n_grid=700,
    is_fitted=True,
    post_dens=None,
    kde=False,
):
    if kde:
        bayes_score = KDE_HPDScore(
            post_estim,
            is_fitted=is_fitted,
            cuda=device == "cuda",
            density_obj=post_dens,
        )
        bayes_score.fit(X_train, theta_train)
    else:
        # using HPDscore to compute posterior probabilities
        bayes_score = HPDScore(
            post_estim,
            is_fitted=is_fitted,
            cuda=device == "cuda",
            density_obj=post_dens,
        )
        bayes_score.fit(X_train, theta_train)

    # first, computing the probability of each observed samples
    prob_array_calib = -bayes_score.compute(X_calib, thetas_calib)

    # computing log_prob for generated samples
    print("Computing log probabilities for sampled data from posterior")
    prob_array_sampled = np.zeros((X_calib.shape[0], B_recal))
    i = 0
    for X in X_calib:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        theta_pos = bayes_score.posterior.sample(
            (B_recal,),
            x=X.to(device if device == "cuda" else "cpu"),
            show_progress_bars=False,
        )

        # computing log_prob for each sampled data
        prob_array_sampled[i, :] = -bayes_score.compute(X, theta_pos, one_X=True)
        i += 1

    # Initializing the HDR recalibration object
    chdr_obj = conditional_hdr_recalibration(
        dim_y=thetas_calib.shape[1],
        fhys=prob_array_calib,
        fhyh=prob_array_sampled,
    )

    # performing recalibration in the test set
    prob_array_test = np.zeros((X_test.shape[0], B_recal))
    theta_samples = np.zeros((X_test.shape[0], B_recal, thetas_calib.shape[1]))

    i = 0
    for X in X_test:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if not isinstance(X, torch.Tensor) or X.dtype != torch.float32:
            X = torch.tensor(X, dtype=torch.float32)

        theta_pos = (
            bayes_score.posterior.sample(
                (B_recal,),
                x=X.to(device if device == "cuda" else "cpu"),
                show_progress_bars=False,
            )
            .cpu()
            .numpy()
        )
        theta_samples[i, :, :] = theta_pos

        prob_array_test[i, :] = -bayes_score.compute(X, theta_pos, one_X=True)
        i += 1

    # recalibrating the samples
    recal_samples, dens_recal = chdr_obj.recal_sample(
        y_hat=theta_samples,
        f_hat_y_hat=prob_array_test,
    )

    target_coverage = 1 - alpha
    cutoff_array = np.zeros(recal_samples.shape[0])

    if score_type == "WALDO":
        mean_list, inv_matrix_list = [], []

    # using the recal samples to compute each cutoff
    for i in tqdm(range(dens_recal.shape[0]), desc="Computing all cutoffs using HDR: "):
        # computing the conformal scores for each case
        if score_type == "HPD":
            conf_scores = -dens_recal[i, :]

        elif score_type == "WALDO":
            conf_scores = np.zeros(dens_recal.shape[1])
            samples = recal_samples[i, :, :]

            # computing waldo stat using the recalibrated samples
            # computing mean and covariance matrix
            mean_array = np.mean(samples, axis=0)
            covariance_matrix = np.cov(samples, rowvar=False)

            if mean_array.shape[0] > 1:
                inv_matrix = np.linalg.inv(covariance_matrix)
            else:
                inv_matrix = 1 / covariance_matrix

            # computing waldo scores for each estimated posterior sample
            for j in range(samples.shape[0]):
                if mean_array.shape[0] > 1:
                    sample_fixed = samples[j, :]
                    conf_scores[j] = (
                        (mean_array - sample_fixed).transpose()
                        @ inv_matrix
                        @ (mean_array - sample_fixed)
                    )
                else:
                    sample_fixed = samples[j]
                    conf_scores[j] = (mean_array - samples[i]) ** 2 / (
                        covariance_matrix
                    )

            mean_list.append(mean_array)
            inv_matrix_list.append(inv_matrix)

        # setting the grid to find the cutoff
        if n_grid is None:
            t_grid = np.arange(
                np.min(conf_scores),
                np.max(conf_scores),
                grid_step,
            )
        else:
            t_grid = np.linspace(
                np.min(conf_scores),
                np.max(conf_scores),
                num=n_grid,
            )

        # computing MC integral for all t_grid
        coverage_array = np.zeros(t_grid.shape[0])
        for t in t_grid:
            coverage_array[t_grid == t] = np.mean(conf_scores <= t)

        closest_t_index = np.argmin(np.abs(coverage_array - target_coverage))
        # finally, finding the naive cutoff
        cutoff_array[i] = t_grid[closest_t_index]

    if score_type == "WALDO":
        return cutoff_array, chdr_obj, mean_list, inv_matrix_list
    else:
        return cutoff_array, chdr_obj
