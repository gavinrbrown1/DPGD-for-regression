# Gavin Brown, grbrown@cs.washington.edu

# compare with TukeyEM

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tukey import tukey, multiple_regressions

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
    

ps = np.concatenate((np.arange(5,50,5),
                     np.arange(50,110,10)))#,
#                      np.arange(150,500,50)))
#                     np.arange(25,205,50)))#,
#                      np.arange(250,1050,50)))

eps = 1
delta = 1e-6

ratio = 2000
trials = 3

OLS_errors = np.zeros((len(ps), trials)) 
GD_errors = np.zeros((len(ps), trials))
SSP_errors = np.zeros((len(ps), trials))
tukey_errors = np.zeros((len(ps), trials))

np.random.seed(345)

for i, p in enumerate(ps):
    n = ratio * p
    print(str(i+1)+'/'+str(len(ps)))
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
        T = 20
        eta = 1/5
        gamma = 5 * math.sqrt(p)
        thetas, fraction_clipped = DP_GD(X, y, T, eps, delta, eta, gamma)
        thetaGD = thetas[-1, :]
        GD_errors[i, trial_num] = np.linalg.norm(theta_true - thetaGD)
        
        # run Tukey 
        models = multiple_regressions(X, y, math.floor(0.95*n/p), use_lasso=False)
        thetaTukey = tukey(models, eps, delta)
        if type(thetaTukey) is not str:
            tukey_errors[i, trial_num] = np.linalg.norm(theta_true - thetaTukey)
        else:
            tukey_errors[i, trial_num] = float('nan')
            print('tukey nan')
            
plt.plot(ps, np.sum(OLS_errors,axis=1)/trials, label='avg OLS error')
plt.plot(ps, np.sum(SSP_errors,axis=1)/trials, label='avg SSP error')
plt.plot(ps, np.sum(GD_errors,axis=1)/trials, label='avg DP-GD error')
plt.plot(ps, np.sum(tukey_errors,axis=1)/trials, label='avg Tukey error')

plt.xlabel(r'Dimension $p$')
plt.ylabel(r'$\ell_2$ error')
plt.title(r'Error when $n='+str(ratio)+'p$, across '+str(trials)+' trials')
plt.legend()
plt.show()
