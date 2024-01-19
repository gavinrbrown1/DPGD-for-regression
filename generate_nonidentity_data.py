# Gavin Brown, grbrown@cs.washington.edu

# code to generate regression data

import numpy as np
from scipy.stats import special_ortho_group

def make_data(n, p):

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

make_data(100, 5)
