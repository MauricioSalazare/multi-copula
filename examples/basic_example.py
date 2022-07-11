from multicopula import EllipticalCopula
import numpy as np

#%%
n_samples_ = 5000
covariance_ = np.array([[   1, -0.6,  0.7],
                        [-0.6,    1, -0.4],
                        [ 0.7,  -0.4,   1]])
mean_ = np.array([1, 3, 4])
data = np.random.multivariate_normal(mean_, covariance_, 5000).T

#%%
copula_model = EllipticalCopula(data)
copula_model.fit()

#%%
samples_ = copula_model.sample(500)
covariance_samples = np.corrcoef(samples_)
