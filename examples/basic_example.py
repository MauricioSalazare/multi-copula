from multicopula import EllipticalCopula
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

#%% Generate random data from a multivariate gaussian of 3-dimensions
n_samples_ = 5000
covariance_ = np.array([[   1, -0.6,  0.7],
                        [-0.6,    1, -0.4],
                        [ 0.7,  -0.4,   1]])
mean_ = np.array([1, 3, 4])
data = np.random.multivariate_normal(mean_, covariance_, 5000).T

#%% Fit the copula model (rows are variables and columns are data samples (instances) of the variables)
copula_model = EllipticalCopula(data)
copula_model.fit()

#%% Sample the copula model
samples_ = copula_model.sample(500)
covariance_samples = np.corrcoef(samples_)

#%% Condition the model in the third dimension
samples_cond1 = copula_model.sample(500, conditional=True, variables={'x3': 3.4})

#%% Condition the model in the second and third dimension
samples_cond2 = copula_model.sample(500, conditional=True, variables={'x2': 2.8, 'x3': 3.4})
