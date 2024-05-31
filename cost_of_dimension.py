import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from time import time
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

def make_data(n, p, isotropic=False):
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
    if isotropic:
        # make covariates
        X = np.random.normal(loc=0, scale=1, size=(n,p))
    
    else: #anisotropic
        # max eigenvalue 2, min 1, rest uniform in that range
        eigvals = np.random.uniform(1, 2, p)
        eigvals[0] = 1
        eigvals[1] = 2

        diagonal_cov = np.diag(eigvals)

        # apply random rotation
        random_matrix = np.random.randn(p, p)
        q, _ = np.linalg.qr(random_matrix)  # QR decomposition for an orthogonal matrix

        # Apply the rotation: Covariance matrix = QDQ^T
        cov = q @ diagonal_cov @ q.T

        # generate covariates
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

    # generate true regression parameter
    theta_true = np.random.normal(size=p)
    theta_true /= np.linalg.norm(theta_true)

    # generate labels
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
    lamb =  max(0, eta - lamb_priv) ##### CHANGE THIS BACK!!! #####
    
    # compute private covariance term
    cov_noise = np.random.normal(loc=0, scale=1, size=(p,p))
    cov_noise[np.tril_indices(p, k=-1)] = cov_noise.T[np.tril_indices(p, k=-1)]
    cov_noise *= math.sqrt(logsod) * Xbd2 / (eps/3)
    priv_cov = cov + cov_noise
    
    # compute private mean term
    mean_noise = np.random.normal(loc=0, scale=1, size=p)
    mean_noise *= math.sqrt(logsod) * Xbd * ybd / (eps/3)
    priv_mean = X.T @ y + mean_noise        ##### CHANGE THIS BACK!!! #####
    
    # combine for final estimate
    theta_priv = np.linalg.inv(priv_cov + lamb*np.eye(p)) @ priv_mean
    return theta_priv

def average_error(algname, algparams, n, p, num_trials):
    """simply report average ell_2 error over num_trials independent runs"""
    errors = np.zeros(num_trials)   # average this later 
    
    # iterate
    for trial_num in range(num_trials):
        X, y, theta_true = make_data(n, p, isotropic=True)
        
        if algname == 'OLS':
            theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        elif algname == 'DPGD':
            rho = algparams['rho']
            T = algparams['T']
            eta = algparams['eta']
            gamma = algparams['gamma']
            
            thetas, fraction_clipped = DP_GD(X, y, T, rho, eta, gamma)
            theta_hat = thetas[-1, :]
        elif algname == 'SSP':
            Xbd = np.max(np.linalg.norm(X, axis=1))
            ybd = np.max(np.abs(y))
            
            eps, delta = algparams['eps'], algparams['delta']
            
            theta_hat = AdaSSP(X, y, eps, delta, Xbd, ybd)
        errors[trial_num] = np.linalg.norm(theta_true - theta_hat)
    return np.mean(errors)
        
def find_target_error(algname, algparams, p, target_error, error_tol, num_trials, verbose=False):
    """
    Find the n that (approximately, on average) achieves the target error.
    Use binary search ("informed" by previous experiments).
    """
    # The code below sets the range for binary search depending on the algorithm.
    # The bounds are a rough guess from previous experiments.
    # The ranges may need to be adjusted if parameters change.
    if algname == 'OLS':
        hi_n = 30*p
        lo_n = p+1
    elif algname == 'DPGD':
        mid = 138*p + 4724
        hi_n = mid + 5000
        lo_n = max(mid - 5000, 0)
    elif algname == 'SSP':
        mid = int(25*math.pow(p, 3/2) + 177*p + 492)
        hi_n = mid + 5000
        lo_n = max(mid - 7000, 0)
    
    for i in range(50):
        guess_n = (hi_n + lo_n) // 2
        error = average_error(algname, algparams, guess_n, p, num_trials)
        
        if verbose:
            print('hi:', hi_n, ' lo:', lo_n, ' guess:', guess_n)
            print('error:', error)
        
        if abs(error - target_error) <= error_tol:
            if verbose:
                print('just right!')
            return guess_n, i
        elif error > target_error:
            if verbose:
                print('too high!')
            lo_n = guess_n
        elif error < target_error:
            if verbose:
                print('too low!')
            hi_n = guess_n
        
        if verbose:
            print()
    
    
###################################
# run the experiment
###################################

# plot iso-error lines for OLS, DP-GD, and AdaSSP
# high privacy

# experimental setup
np.random.seed(123)
num_trials = 50
target_error = 0.5
error_tol = 0.01

ps = np.array([5*i for i in range(1,21)])
target_ns = np.zeros((3, len(ps)))

total_start = time()
print('Running for OLS')
for i, p in enumerate(ps):
    start = time()
    algparams = {}
    target_ns[0,i], search_calls = find_target_error('OLS', algparams, p, target_error, error_tol, num_trials)
    end = time()
    print(p, ' : ', search_calls, ' : ', end-start)

print('Running for DP-GD')
for i, p in enumerate(ps):
    start = time()
    algparams = {'rho': 0.015,
                'T': 20,
                'eta': 1/5,
                'gamma': 5*math.sqrt(p)}
    target_ns[1,i], search_calls = find_target_error('DPGD', algparams, p, target_error, error_tol, num_trials)
    end = time()
    print(p, ' : ', search_calls, ' : ', end-start)

print('Running for AdaSSP')
for i, p in enumerate(ps):
    start = time()
    algparams = {'eps': 0.925,
                'delta': 1e-6}
    target_ns[2,i], search_calls = find_target_error('SSP', algparams, p, target_error, error_tol, num_trials)
    end = time()
    print(p, ' : ', search_calls, ' : ', end-start)

total_end = time()
print('total time:', total_end-total_start)

###################################
# fit curves to the observations
###################################

# try a p^{3/2}+p fit for SSP
def model_func(x, a, b, c):
    return a * x**(3/2) + b * x + c

# Fit the model to the data
popt, pcov = curve_fit(model_func, ps, target_ns[2,:])

# Extract the optimal parameters
a, b, c = popt
print(f'Optimal parameters: a = {a}, b = {b}, c = {c}')

# Calculate the fitted values
y_fit_SSP = model_func(ps, *popt)

# try a p fit for DP-GD
def model_func(x, a, b):
    return a * x + b

# Fit the model to the data
popt, pcov = curve_fit(model_func, ps, target_ns[1,:])

# Extract the optimal parameters
a, b = popt
print(f'Optimal parameters: a = {a}, b = {b}')

# Calculate the fitted values
y_fit_DPGD = model_func(ps, *popt)

###################################
# plots!
###################################

plt.plot(ps, target_ns[2,:], '-o', label='AdaSSP')
plt.plot(ps, y_fit_SSP, '--', color='black', label=r'AdaSSP fit $\approx 24 p^{3/2} + 189p + O(1)$')
plt.plot(ps, target_ns[1,:], '-o', label='DP-GD')
plt.plot(ps, y_fit_DPGD, '-.', color='black', label=r'DP-GD fit $\approx 177 p + O(1)$')
plt.plot(ps, target_ns[0,:], '-o', label='OLS')
plt.xlabel('Dimension')
plt.ylabel(r'Samples Needed for $\mathbb{E}\lVert \tilde\theta-\theta*\rVert_2 \approx 0.5$')
plt.text(25-2, 1400, '210')
plt.text(50-2, 1600, '416')
plt.text(75-2, 1700, '677')
plt.text(100-2, 2000, '871')
plt.legend()
plt.show()
