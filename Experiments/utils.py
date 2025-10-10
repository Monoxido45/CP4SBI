import torch
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from sbi.inference import SNPE
from sbi.inference import simulate_for_sbi
from sbi.utils import BoxUniform
from tqdm import tqdm


def train_sbi_amortized(
    sim_budget,
    simulator,
    prior,
    density_estimator='nsf',
    save_fname=None,
    return_density = False,
    nuisance = False,
):
    if not nuisance:
        theta, x = simulate_for_sbi(
            simulator, proposal=prior, num_simulations=sim_budget)
        simulations = {}
        simulations['theta'] = theta
        simulations['x'] = x

        # create inference object. Here, NPE is used.
        inference = SNPE(prior=prior, density_estimator=density_estimator)

        inference = inference.append_simulations(
            simulations['theta'], simulations['x'], proposal=prior
        )

        # train the density estimator and build the posterior
        density_approx = inference.train(
            stop_after_epochs=20,
            max_num_epochs=250,
            training_batch_size=100
        )

        if save_fname is not None:
            # saving the results into a file
            parameters_approx = inference._neural_net.state_dict()
            torch.save(parameters_approx, save_fname)
        if return_density:
            return [density_approx, inference]
        else:
            posterior = inference.build_posterior(density_approx)
            return posterior
    else:
        prior_list = [
            BoxUniform(
            low=torch.tensor([10.0, 50.0]),
            high=torch.tensor([250.0, 500.0])
                ),
            BoxUniform(
            low=torch.tensor([10.0, 100.0]),
            high=torch.tensor([250.0, 5000.0])),
            BoxUniform(
            low=torch.tensor([50.0, 100.0]),
            high=torch.tensor([500.0, 5000.0])
            )
        ]
        comb_list = [[0, 1], [0, 2], [1, 2]]
        dens_list, inf_list = [], []

        # simulating only one time
        theta, x = simulate_for_sbi(
        simulator, proposal=prior, num_simulations=sim_budget)
        simulations = {}
        simulations['theta'] = theta
        simulations['x'] = x

        # fitting three neural networks, one for each parameter combination
        for pars, prior_sel in tqdm(zip(comb_list, prior_list), desc = 'Training for different parameter combinations'):
            thetas_used = simulations['theta'][:, pars]
            X_used = simulations['x']

            # create inference object. Here, NPE is used.
            inference_obj = SNPE(prior=prior_sel, density_estimator=density_estimator)

            inference_obj = inference_obj.append_simulations(
            thetas_used, X_used, proposal=prior_sel
            )

            # train the density estimator and build the posterior
            density_approx_obj = inference_obj.train(
            stop_after_epochs=20,
            max_num_epochs=250,
            training_batch_size=100
            )

            dens_list.append(density_approx_obj)
            inf_list.append(inference_obj)

            if save_fname is not None:
                # saving the results into a file
                parameters_approx = inference._neural_net.state_dict()
                torch.save(parameters_approx, save_fname + f'_pars_{pars[0]}{pars[1]}.pkl')
            return dens_list, inf_list


def train_sbi_multiround(
    x_obs,
    sim_budget_list,
    simulator,
    prior,
    density_estimator='nsf',
    save_fname=None,
):

    num_rounds = len(sim_budget_list)

    # create inference object. Here, NPE is used.
    inference = SNPE(prior=prior, density_estimator=density_estimator)

    posterior_list = []
    proposal = prior

    for r in range(num_rounds):

        theta, x = simulate_for_sbi(
            simulator, proposal=prior, num_simulations=sim_budget_list[0])
        simulations = {}
        simulations['theta'] = theta
        simulations['x'] = x

        inference = inference.append_simulations(
            simulations['theta'], simulations['x'], proposal=proposal
        )

        # train the density estimator and build the posterior
        density_approx = inference.train(
            stop_after_epochs=20,
            max_num_epochs=250,
            training_batch_size=100
        )
        posterior = inference.build_posterior(density_approx)

        if save_fname is not None:
            # saving the results into a file
            parameters_approx = inference._neural_net.state_dict()
            sv = save_fname.replace('.pkl', '')
            sv = sv + f'_round_{r:02}.pkl'
            torch.save(parameters_approx, sv)

        posterior_list.append(posterior)
        proposal = posterior.set_default_x(x_obs)

    return posterior_list


def get_posterior_approx(
    load_fname,
    prior,
    simulator,
    density_estimator='nsf'
):

    # build the posterior object
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    build_nn_posterior = posterior_nn(model=density_estimator)
    batch_theta = prior.sample((2,))
    batch_x = simulator(batch_theta)
    inference._neural_net = build_nn_posterior(batch_theta, batch_x)
    inference._neural_net.load_state_dict(torch.load(load_fname))
    inference._neural_net.eval()
    posterior = inference.build_posterior(inference._neural_net)

    return posterior
