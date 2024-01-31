# show "cost of privacy" for fixed dimension as data size grows

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import time
from scipy.optimize import curve_fit

def make_data(n, p):
    """
        Make a regression problem.
        Input
            n: number of samples
            p: dimension
        Output
            X: covariates
            y: responses
            theta_true: true parameter
    """
    # true regression parameter
    theta_true = np.random.normal(size=p)
    theta_true /= np.linalg.norm(theta_true)
    
    # make data
    X = np.random.normal(loc=0, scale=1, size=(n,p))
    y = X @ theta_true 
    y += np.random.normal(loc=1, scale=1, size=n)
    
    return X, y, theta_true


def DP_GD(X, y, T, rho, eta, gamma, verbose=False):
    """
        Differentially private gradient descent.
        Input
            X: numpy shape (n,p)
            y: numpy shape (n,)
            T: int, number of iterations
            rho: float, privacy param
            eta: float, step size
            gamma: float, clipping threshold
        Output
            thetas: parameter estimates, (T+1,p)
            fraction_clipped: what fraction of gradients were clipped?
    """
    # set up
    n, p = X.shape
    variance = 2 * T * math.pow(gamma,2) / (rho * n**2)
    thetas = np.zeros((T+1, p))

    total_clipped = 0 
    
    # iterate gradient descent
    for t in range(1, T+1):
        # compute residuals
        residuals = y - X @ thetas[t-1, :]
        residuals = np.reshape(residuals, (n,1))

        # compute per-point gradient
        gradients = X * residuals    # this has shape (n,p)

        # clip to gamma
        norms = np.linalg.norm(gradients, axis=1)
        violated = (norms > gamma)
        num_violated = np.sum(violated)
        if verbose:
            print('Number gradients clipped:', num_violated)
        if num_violated > 0:
            gradients[violated, :] = gamma * gradients[violated, :] / norms[violated].reshape((num_violated,1))
            total_clipped += num_violated

        # average
        average_gradient = np.sum(gradients, axis=0) / n

        # draw noise and step
        noise = math.sqrt(variance) * np.random.normal(loc=0, scale=1, size=p)
        thetas[t,:] = thetas[t-1,:] + eta * (average_gradient + noise)

    return thetas, (total_clipped / (T*n))


def cost_of_privacy_experiment(rho, p, nmin, nmax, nstep, num_trials):
    ns = np.arange(nmin,nmax,nstep)

    OLS_errors = np.zeros((len(ns), num_trials)) 
    GD_errors = np.zeros((len(ns), num_trials))

    print('Running experiment with rho='+str(rho))
    start = time()
    for i, n in enumerate(ns):
        for trial_num in range(num_trials):
            # generate data
            X, y, theta_true = make_data(n, p)

            # compute nonprivate OLS and its error (from true theta)
            thetaOLS = np.linalg.inv(X.T @ X) @ X.T @ y
            OLS_errors[i, trial_num] = np.linalg.norm(theta_true - thetaOLS)

            # run DP-GD (error from empirical theta)
            T = 20
            eta = 1/5
            gamma = 3 * math.sqrt(p)
            thetas, fraction_clipped = DP_GD(X, y, T, rho, eta, gamma)
            thetaGD = thetas[-1, :]
            GD_errors[i, trial_num] = np.linalg.norm(thetaOLS - thetaGD)  
    end = time()
    print('Time:', end-start)
    
    plt.plot(ns, np.average(OLS_errors, axis=1), label=r'Sampling Noise, $||\theta_{OLS}-\theta^*||$')
    plt.plot(ns, np.average(GD_errors, axis=1), label=r'Privacy Noise, $||\tilde{\theta}-\theta_{OLS}||$')

    # Fit 1/sqrt(n) function
    def fit_func(x, a, b):
        return a / np.sqrt(x) + b
    params, params_covariance = curve_fit(fit_func, ns, np.average(OLS_errors, axis=1), p0=[1, 0])
    plt.plot(ns, fit_func(ns, *params), 'k:', label='Inverse Square Root Fit')

    # plot 1/n function
    def fit_func(x, a, b):
        return a / x + b
    params, params_covariance = curve_fit(fit_func, ns, np.average(GD_errors, axis=1), p0=[1, 0])
    plt.plot(ns, fit_func(ns, *params), 'k--', label='Inverse Linear Fit')

    plt.grid(linewidth=0.5, color='lightgray')
    plt.title(r'Errors from Privacy and Sampling, $p='+str(p)+r'$, $\rho='+str(rho)+'$')
    plt.xlabel(r'Number of Samples, $n$')
    plt.ylabel(r'Average $\ell_2$ error')
    plt.legend()
    plt.show()

############################################################
##### run the experiments
############################################################

np.random.seed(345)

# low privacy
cost_of_privacy_experiment(rho=0.1,
                          p=10,
                          nmin=1000,
                          nmax=5100,
                          nstep=100,
                          num_trials=100)

# high privacy
cost_of_privacy_experiment(rho=0.015,
                          p=10,
                          nmin=10000,
                          nmax=36000,
                          nstep=1000,
                          num_trials=100)

