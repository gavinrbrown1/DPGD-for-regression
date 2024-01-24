# show "cost of privacy" for fixed dimension as data size grows

trials = 50

p = 10
ns = np.arange(100,1000,50)

OLS_errors = np.zeros((len(ns), trials)) 
GD_errors = np.zeros((len(ns), trials))
# SSP_errors = np.zeros((len(ps), trials))

eps = 2
delta = 1e-6

np.random.seed(345)

for i, n in enumerate(ns):
    for trial_num in range(trials):
        # generate data
        X, y, theta_true = make_data(n, p)
        
        # compute nonprivate OLS and its error (from true theta)
        thetaOLS = np.linalg.inv(X.T @ X) @ X.T @ y
        OLS_errors[i, trial_num] = np.linalg.norm(theta_true - thetaOLS)
        
        # run DP-GD (error from empirical theta)
        T = 20
        eta = 1/5
        gamma = 3 * math.sqrt(p)
        thetas = DP_GD(X, y, T, eps, delta, eta, gamma)
        thetaGD = thetas[-1, :]
        GD_errors[i, trial_num] = np.linalg.norm(thetaOLS - thetaGD)
        
plt.plot(ns, np.average(OLS_errors, axis=1), label=r'Sampling Noise, $||\theta_{OLS}-\theta^*||$')
plt.plot(ns, np.average(GD_errors, axis=1), label=r'Privacy Noise, $||\tilde{\theta}-\theta_{OLS}||$')
# plt.plot(ns, np.zeros(len(ns)), 'k--')

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

plt.title('Privacy and Sampling Error, $p=10$')
plt.xlabel(r'Number of Samples, $n$')
plt.ylabel(r'Average $\ell_2$ error')
plt.legend()
plt.show()
