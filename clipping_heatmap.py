# Gavin Brown, grbrown@cs.washington.edu

# make a heatmap showing can clip at square-root level

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# test out clipping percentage as a function of gamma

np.random.seed(123)
eps = 1
delta = 1e-6

ps = np.arange(start=5, stop=205, step=5)
gammas = np.linspace(math.sqrt(ps[0]), 3*math.sqrt(ps[-1]), 20)
clippings = np.zeros((len(ps),len(gammas)))

np.random.seed(123)
eps = 1
delta = 1e-6

ps = np.arange(start=5, stop=205, step=5)
gammas = np.linspace(math.sqrt(ps[0]), 3*math.sqrt(ps[-1]), 20)
clippings = np.zeros((len(ps),len(gammas)))

for i, p in enumerate(ps):
    n = 100*p
    print(p) 
    for j, gamma in enumerate(gammas):
        X, y, theta_true = make_data(n, p)

        # DP-GD
        T = 20
        eta = 1/5
        thetas, fraction_clipped = DP_GD(X, y, T, eps, delta, eta, gamma, verbose=False)
        clippings[i,j] += fraction_clipped

fig, ax = plt.subplots()
cax = ax.imshow(clippings.T[::-1,:], cmap='viridis', aspect='auto')

# plot square root line
x_values = np.linspace(0, len(ps)-1, 100)  # Increase 300 for smoother line

# Calculate corresponding y-values using y = sqrt(x)
# this is just a function that "looks right," found by tweaking
y_values = - 2.5*np.sqrt(x_values) + 19

plt.plot(x_values, y_values, 'r--')  # 'r--' for red dashed line

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

plt.xticks(range(len(ps))[::5], ps[::5])  # Set x-axis labels
plt.yticks(range(len(gammas))[::2], ['{:.1f}'.format(tick) for tick in gammas[::-1]][::2])  # Set y-axis labels

plt.text(5, 9.5, r'$f(x) \propto \sqrt{x}$', fontsize=12, color='red')

plt.colorbar(cax)
plt.xlabel(r'dimension $p$')
plt.ylabel(r'clipping threshold $\gamma$')
plt.title('Fraction of Gradients Clipped')

# Show the plot
plt.show()


