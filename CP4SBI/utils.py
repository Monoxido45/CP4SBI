import numpy as np

# defining naive function
def naive_method(
        post_estim,
        X,
        alpha = 0.1,
        score_type = "HPD",
        B_naive = 1000,
        device = "cuda",
        B_waldo = 1000,
        grid_step = 0.005,
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
    
    # samples to compute scores
    samples = post_estim.sample(
        (B_naive,),
        x=X.reshape(1, -1),
        show_progress_bars=False,
    )
    
    # if score_type is HPD
    if score_type == "HPD":
        conf_scores = -np.exp(
            post_estim.log_prob_batched(
                samples,
                x=X.reshape(1, -1),
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
                x=X.reshape(1,-1),
                show_progress_bars=False,
            )
            .cpu()
            .detach()
            .numpy()   
        )

        print(sample_generated.shape)
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
    t_grid = np.arange(
        np.min(conf_scores),
        np.max(conf_scores),
        grid_step,
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
