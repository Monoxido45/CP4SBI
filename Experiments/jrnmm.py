import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from numba import njit


@njit
def get_jrnmm_params():
    """
    Returns the fixed parameters for the Jansen-Rit Neural Mass Model (JRNMM).

    The parameters returned by this function are based on the original Jansen-Rit 
    paper, which describes a neural mass model of the electroencephalogram (EEG).

    Returns:
        tuple: A tuple containing the following parameters:
            - s4 (float): Parameter s4, typically set to 0.01.
            - s6 (float): Parameter s6, typically set to 1.00.
            - A (float): Average excitatory synaptic gain, typically set to 3.25.
            - a (float): Inverse of the time constant of excitatory synapses, typically set to 100.0.
            - B (float): Average inhibitory synaptic gain, typically set to 22.0.
            - b (float): Inverse of the time constant of inhibitory synapses, typically set to 50.0.
            - vmax (float): Maximum firing rate, typically set to 5.0.
            - r (float): Sigmoid slope parameter, typically set to 0.56.
            - v0 (float): Sigmoid threshold parameter, typically set to 6.0.

    Reference:
        Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked potential 
        generation in a mathematical model of coupled cortical columns. Biological Cybernetics, 
        73(4), 357-366.
    """

    A = 3.25
    B = 22.0
    a = 100.0
    b = 50.0
    vmax = 5.0
    v0 = 6.0
    r = 0.56
    s4 = 0.01
    s6 = 1.00
    return s4, s6, A, a, B, b, vmax, r, v0


@njit
def sigmoid(x, vmax, r, v0):
    """
    Compute the sigmoid function.

    The sigmoid function is defined as:
        sigmoid(x) = vmax / (1 + exp(r * (v0 - x)))

    Parameters:
    x (float): The input value.
    vmax (float): The maximum value of the sigmoid function.
    r (float): The steepness of the curve.
    v0 (float): The value of x at the sigmoid's midpoint.

    Returns:
    float: The computed sigmoid value.
    """
    num = vmax
    den = 1.0 + np.exp(r * (v0 - x))
    return num / den


@njit
def G(q, mu, sigma, C, A, a, B, b, vmax, r, v0):
    """
    Compute the G vector based on the given parameters.

    Parameters:
    q (array-like): Input vector with at least 3 elements.
    mu (float): Mean value parameter.
    sigma (float): Standard deviation parameter.
    C (float): Scaling constant.
    A (float): Parameter for the first term.
    a (float): Parameter for the first term.
    B (float): Parameter for the third term.
    b (float): Parameter for the third term.
    vmax (float): Maximum value for the sigmoid function.
    r (float): Steepness parameter for the sigmoid function.
    v0 (float): Midpoint value for the sigmoid function.

    Returns:
    np.ndarray: Array containing the computed G1, G2, and G3 values.
    """

    C1 = C
    C2 = 0.80*C
    C3 = 0.25*C
    C4 = 0.25*C

    G1 = A * a * sigmoid(q[1] - q[2], vmax, r, v0)
    G2 = A * a * (mu + C2 * sigmoid(C1 * q[0], vmax, r, v0))
    G3 = B * b * (C4 * sigmoid(C3 * q[0], vmax, r, v0))

    return np.array([G1, G2, G3])


@njit
def get_expAt(t, a, b):
    """
    Compute the matrix exponential for given parameters.

    This function calculates a specific matrix exponential based on the input
    parameters `t`, `a`, and `b`. The resulting matrix is a 6x6 diagonal matrix
    with additional off-diagonal elements set according to the given formula.

    Parameters:
    t (float): The time parameter.
    a (float): The first rate parameter.
    b (float): The second rate parameter.

    Returns:
    numpy.ndarray: A 6x6 matrix representing the computed matrix exponential.
    """

    eat = np.exp(-a*t)
    eatt = np.exp(-a*t) * t
    ebt = np.exp(-b*t)
    ebtt = np.exp(-b*t) * t
    expAt = np.diag(np.array(
       [eat+a*eatt,
        eat+a*eatt,
        ebt+b*ebtt,
        eat-a*eatt,
        eat-a*eatt,
        ebt-b*ebtt
        ]))
    expAt[0, 3] = eatt
    expAt[1, 4] = eatt
    expAt[2, 5] = ebtt
    expAt[3, 0] = -a**2 * eatt
    expAt[4, 1] = -a**2 * eatt
    expAt[5, 2] = -b**2 * ebtt

    return expAt


@njit
def get_covariance(delta, mu, sigma, C, A, a, B, b, s4, s6):
    """
    Calculate the covariance matrix for given parameters.

    Parameters:
    delta (float): Time increment.
    mu (float): Mean parameter (not used in the current implementation).
    sigma (float): Standard deviation parameter.
    C (float): Constant parameter (not used in the current implementation).
    A (float): Constant parameter (not used in the current implementation).
    a (float): Decay rate parameter for the first and second elements.
    B (float): Constant parameter (not used in the current implementation).
    b (float): Decay rate parameter for the third element.
    s4 (float): Standard deviation for the first element.
    s6 (float): Standard deviation for the third element.

    Returns:
    numpy.ndarray: A 6x6 covariance matrix.
    """

    em2at = np.exp(-2*a*delta)
    em2bt = np.exp(-2*b*delta)
    e2at = np.exp(2*a*delta)
    e2bt = np.exp(2*b*delta)
    sg = np.array([s4, sigma, s6])**2

    cov = np.diag(np.array([
       em2at*(e2at-1-2*a*delta*(1+a*delta))*sg[0]/(4*a**3),
       em2at*(e2at-1-2*a*delta*(1+a*delta))*sg[1]/(4*a**3),
       em2bt*(e2bt-1-2*b*delta*(1+b*delta))*sg[2]/(4*b**3),
       em2at*(e2at-1-2*a*delta*(a*delta-1))*sg[0]/(4*a),
       em2at*(e2at-1-2*a*delta*(a*delta-1))*sg[1]/(4*a),
       em2bt*(e2bt-1-2*b*delta*(b*delta-1))*sg[2]/(4*b)
    ]))

    cov[0, 3] = em2at*(delta**2)*sg[0]/2
    cov[1, 4] = em2at*(delta**2)*sg[1]/2
    cov[2, 5] = em2bt*(delta**2)*sg[2]/2
    cov[3, 0] = em2at*(delta**2)*sg[0]/2
    cov[4, 1] = em2at*(delta**2)*sg[1]/2
    cov[5, 2] = em2bt*(delta**2)*sg[2]/2

    return cov


@njit
def simulate_jrnmm(mu, sigma, C, tarray, burnin, downsample=1):
    """
    Simulate the stochastic Jansen-Rit neural mass model (JRNMM) using the Strang-splitting method.

    This function implements the Strang-splitting method for the stochastic Jansen-Rit neural mass model
    as described in the literature. The method is preferred over the Euler-Maruyama method for its ability
    to better preserve the statistical behavior of the generated time series, even with a relatively large
    time step (dt).

    Parameters
    mu : float
        Mean of the background activity coming from neighboring columns.
    sigma : float
        Standard deviation of the background activity coming from neighboring columns.
    C : array_like
        Internal connectivity matrix of the different parts of the cortical column.
    tarray : array_like
        Array of time points at which to simulate the model.
    burnin : float
        Time to discard at the beginning of the simulation to allow the system to reach a steady state.
    downsample : int, optional
        Factor by which to downsample the output time series. Default is 1 (no downsampling).

    Returns
    -------
    Y : numpy.ndarray
        Simulated time series of the difference between the second and third state variables.
    t : numpy.ndarray
        Array of time points corresponding to the downsampled time series.

    [1] Ableidinger et al. "A Stochastic Version of the Jansen and Rit Neural Mass Model: Analysis and Numerics" (2017)
    [2] Buckwar et al. "Spectral density-based and measure-preserving ABC for partially observed diffusion processes. 
        An illustration on Hamiltonian SDEs" (2020)
    """

    # get the jrnmm fixed parameters
    s4, s6, A, a, B, b, vmax, r, v0 = get_jrnmm_params()

    # supposing that the delta is fixed
    delta = tarray[1] - tarray[0]

    # pre-calculate the exp(A*t) factor that is used to solve the ODE step
    expAdelta = get_expAt(delta, a, b)

    # pre-calculate the covariance matrix of the input noise for the SDE step
    cov = get_covariance(delta, mu, sigma, C, A, a, B, b, s4, s6)

    # generate noise
    noise = np.random.randn(6, len(tarray))
    noise = np.linalg.cholesky(cov) @ noise
    noise = noise.T

    X = np.zeros((len(tarray), 6))

    for i, _ in enumerate(tarray[:-1]):

        Q = X[i][:3]

        # step (1) -- integrate the non-linear ODE part of the model
        perturb = np.zeros(6)
        perturb[3:] = delta/2 * G(Q, mu, sigma, C, A, a, B, b, vmax, r, v0)
        Xb = X[i] + perturb

        # step (2) -- integrate the linear SDE part of the model
        Xa = np.dot(expAdelta, Xb) + noise[i]

        # step (3) -- integrate the non-linear ODE part of the model
        Qa = Xa[:3]
        perturb = np.zeros(6)
        perturb[3:] = delta/2 * G(Qa, mu, sigma, C, A, a, B, b, vmax, r, v0)
        Xprox = Xa + perturb

        X[i+1, :] = Xprox

    Y = X[:, 1] - X[:, 2]

    Y = Y[int(burnin/delta)::downsample]
    t = tarray[:-int(burnin/delta):downsample]

    return Y, t


if __name__ == '__main__':

    # choose values of the JRNMM for the simulation
    mu = 220.0
    sigma = 2000.0
    C = 135.0

    # define timespan
    delta = 1e-3
    burnin = 2  # given in seconds
    duration = 8  # given in seconds
    downsample = 4
    tarray = np.arange(0, burnin + duration, step=delta)

    X_list = []

    for _ in range(10):
        # simulate JRNMM model with Strang splitting
        X, t = simulate_jrnmm(mu, sigma, C, tarray, burnin, downsample)
        X_list.append(X)

    # plot the power spectrum
    fig, ax = plt.subplots(figsize=(12, 5), ncols=2)
    for X in X_list:
        f, Pxx = welch(X, fs=1/(downsample*delta))
        ax[0].plot(t, X, lw=2.0)
        ax[1].plot(f, Pxx, lw=2.0)
        ax[1].set_xlim(0, 50)
    fig.show()
