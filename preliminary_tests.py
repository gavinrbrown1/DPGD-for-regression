# Gavin Brown, grbrown@bu.edu
# Code on DP linear regression

# Feel free to modify and use for any purpose.
# Prototype code: do not rely on to protect privacy.

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time

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

def DP_GD(X, y, T, eps, delta, eta, gamma, verbose=False):
    """
        Differentially private gradient descent.
        Input
            X: numpy shape (n,p)
            y: numpy shape (n,)
            T: int, number of iterations
            eps: float, privacy param
            delta: float, privacy param
            eta: float, step size
            gamma: float, clipping threshold
        Output
            thetas: parameter estimates, (T+1,p)
    """
    # set up
    n, p = X.shape
    variance = T * math.log(1/delta) * math.pow(gamma,2) / (math.pow(eps,2) * n**2)
    thetas = np.zeros((T+1, p))

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

        # average
        average_gradient = np.sum(gradients, axis=0) / n

        # draw noise and step
        noise = math.sqrt(variance) * np.random.normal(loc=0, scale=1, size=p)
        thetas[t,:] = thetas[t-1,:] + eta * (average_gradient + noise)

    return thetas

def AdaSSP(X, y, eps, delta, Xbd, ybd):
    """
        AdaSSP algorithm, a practical standard.
        Mostly copied from available matlab code.
        Input
            X: covariates, shape (n, p)
            y: responses, shape (n,)
            eps,delta: privacy
            Xbd: prior bound on max norm of covariates
            ybd: prior bound on max magnitude of responses
        Output
            theta_priv: estimate, shape (p,)
    """
    # prep work
    cov = X.T @ X
    Xbd2 = math.pow(Xbd,2)
    logsod = math.log(6/delta)    # "log six over delta"
    varrho = 0.05    # what is this? set to in matlab code, is it failure probability?
    eta = Xbd2 * math.sqrt(p * logsod * math.log(2*(p**2)/varrho)) / (eps/3)

    # calculate min eigenvalue
    lamb_min = np.min(np.linalg.eigvalsh(cov))

    # compute private ridge parameter
    noise = np.random.normal()
    lamb_priv = lamb_min + (Xbd2 / (eps/3)) * (noise*math.sqrt(logsod) - logsod)
        # Above line is different between pseudocode and code on github!
        # If I understand, it doesn't affect privacy, but might affect utility.
    lamb_priv = max(lamb_priv, 0)

    # set ridge parameter to use
    lamb = max(0, eta - lamb_priv)

    # compute private covariance term
    cov_noise = np.random.normal(loc=0, scale=1, size=(p,p))
    cov_noise[np.tril_indices(p, k=-1)] = cov_noise.T[np.tril_indices(p, k=-1)]
    cov_noise *= math.sqrt(logsod) * Xbd2 / (eps/3)
    priv_cov = cov + cov_noise

    # compute private mean term
    mean_noise = np.random.normal(loc=0, scale=1, size=p)
    mean_noise *= math.sqrt(logsod) * Xbd * ybd / (eps/3)
    priv_mean = X.T @ y + mean_noise

    # combine for final estimate
    theta_priv = np.linalg.inv(priv_cov + lamb*np.eye(p)) @ priv_mean
    return theta_priv

def CI_from_checkpts(thetas, num_checkpts, burnin, interval, confidence):
    """
        Construct confidence intervals around empirical estimator.
        Input
            thetas: private estimates, shape (T, p)
            num_checkpts: int, how many steps to look at
            burnin: int, when do we take the first checkpt?
            interval: int, after that, checkpt how often?
            confidence: float, confidence level
        Output
            point_estimate: mean, shape (p,)
            upper: top of each CI, shape (p,)
            lower: bottom, shape (p,)
    """
    # setup, z-scores etc
    T, p = thetas.shape
    assert burnin + (num_checkpts-1)*interval < T  # need enough room for all checkpts
    z = norm.ppf(1 - (1-confidence)/2)  # from scipy

    # where are the checkpts?
    # first checkpt is:   thetas[burnin,:]
    # next is:            thetas[burnin+interval,:]
    # last is:            thetas[burnin+interval*(num_checkpts-1),:]
    checkpt_indices = [burnin + i*interval for i in range(num_checkpts)]
    checkpts = thetas[checkpt_indices,:]

    # compute mean of checkpts
    point_estimate = np.mean(checkpts, axis=0)

    # compute per-coordinate variances
    stddevs = np.std(checkpts, axis=0)

    # construct CIs
    scales = z * stddevs / math.sqrt(num_checkpts)
    upper = point_estimate + scales
    lower = point_estimate - scales

    return point_estimate, upper, lower

# Generate some data!

p = 5
n = 500*p

eps = 1
delta = 1e-6

np.random.seed(123)

start = time()
X = np.random.normal(loc=0, scale=1, size=(n,p))

theta_true = np.zeros(p)
theta_true[0] = 1
y = X @ theta_true
y += np.random.normal(loc=0, scale=1, size=n)
end = time()
# print('data gen time:', end-start)

# nonprivate OLS estimate
start = time()
theta_empirical = np.linalg.inv(X.T @ X) @ X.T @ y
empirical_error = np.linalg.norm(theta_true - theta_empirical)
# print('nonprivate error:', np.linalg.norm(theta_true - theta_empirical))
end = time()
# print('nonprivate time:', end-start)

# AdaSSP estimate
Xbd = np.max(np.linalg.norm(X, axis=1))  # this makes it nonprivate, but favorable to AdaSSP.
ybd = np.max(np.abs(y))
print('Xbd:', Xbd)
print('ybd:', ybd)
thetaSSP = AdaSSP(X, y, eps, delta, Xbd, ybd)
SSP_error = np.linalg.norm(theta_true - thetaSSP)

T = 50
eta = 1/5
gamma = 5 * math.sqrt(p)

thetas = DP_GD(X, y, T, eps, delta, eta, gamma, verbose=False)

diffs = np.linalg.norm(thetas - theta_true.reshape((1,p)), axis=1)

plt.plot(np.arange(T+1), diffs, label='DP-GD')
plt.plot(np.arange(T+1), empirical_error*np.ones(T+1), '--', label='Nonprivate')
plt.plot(np.arange(T+1), SSP_error*np.ones(T+1), '--', label='AdaSSP')
plt.xlabel('Gradient Iterations')
plt.ylabel(r'$\ell_2$ distance from $\theta^*$')
plt.legend()
plt.show()

# now make confidence intervals from that
point_estimate, upper, lower = CI_from_checkpts(thetas,
                                                num_checkpts=10,
                                                burnin=5,
                                                interval=5,
                                                confidence=0.9)

print(upper)
print(point_estimate)
print(lower)
print(theta_empirical)
print((theta_empirical >= lower) & (theta_empirical <= upper))

###################################################################################
###################################################################################
###################################################################################
# try across different dimensions, keeping p/n fixed
# (this take several minutes on my macbook)

trials = 5

ps = np.concatenate((np.arange(2,21),
                    np.arange(25,205,5),
                     np.arange(250,1050,50)))
OLS_errors = np.zeros((len(ps), trials))
GD_errors = np.zeros((len(ps), trials))
SSP_errors = np.zeros((len(ps), trials))

eps = 1
delta = 1e-6

ratio = 100

np.random.seed(345)

for i, p in enumerate(ps):
    n = ratio * p
    print(str(i)+'/'+str(len(ps))+', ', end='')
    for trial_num in range(trials):
        # generate data
        X, y, theta_true = make_data(n, p)

        # compute nonprivate OLS and its error
        thetaOLS = np.linalg.inv(X.T @ X) @ X.T @ y
        OLS_errors[i, trial_num] = np.linalg.norm(theta_true - thetaOLS)

        # run AdaSSP, with nonprivate bounds on data
        Xbd = np.max(np.linalg.norm(X, axis=1))
        ybd = np.max(np.abs(y))
        thetaSSP = AdaSSP(X, y, eps, delta, Xbd, ybd)
        SSP_errors[i, trial_num] = np.linalg.norm(theta_true - thetaSSP)

        # run DP-GD
        T = 5
        eta = 1/3
        gamma = 5 * math.sqrt(p)
        thetas = DP_GD(X, y, T, eps, delta, eta, gamma)
        thetaGD = thetas[-1, :]
        GD_errors[i, trial_num] = np.linalg.norm(theta_true - thetaGD)

plt.plot(ps, np.sum(OLS_errors,axis=1)/trials, label='avg OLS error')
plt.plot(ps, np.sum(SSP_errors,axis=1)/trials, label='avg SSP error')
plt.plot(ps, np.sum(GD_errors,axis=1)/trials, label='avg DP-GD error')

plt.text(100, 0.8, 'AdaSSP', color='tab:orange', size=12)
plt.text(20, 0.30, 'DP-GD', color='tab:green', size=12)
plt.text(160, 0.165, 'OLS', color='tab:blue', size=12)

plt.xscale('log')
plt.xlabel(r'Dimension $p$, log scale')
plt.ylabel(r'$\ell_2$ error')
plt.title(r'Error when $n=100p$, across 5 trials')
plt.show()
