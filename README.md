[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MauricioSalazare/multi-copula/master?urlpath=lab/tree/examples)

# MultiCopula

## What is MultiCopula?
 
It is a multivariate probabilistic modelling package, which uses copula theory.

## How to install
The package can be installed via pip using:

```shell
    pip install multicopula
```



## Example:

Run the load base case as:

```python
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

#%% Fit the copula model
copula_model = EllipticalCopula(data)
copula_model.fit()

#%% Sample the copula model
samples_ = copula_model.sample(500)
covariance_samples = np.corrcoef(samples_)

#%% Condition the model in the third dimension
samples_cond1 = copula_model.sample(500, conditional=True, variables={'x3': 3.4})

#%% Condition the model in the second and third dimension
samples_cond2 = copula_model.sample(500, conditional=True, variables={'x2': 2.8, 'x3': 3.4})

```
   
The package focuses in the simulation of daily electrical consumption profiles for low voltage and medium
voltage networks. Example of generated profiles conditioned to a yearly energy consumption [link](https://github.com/MauricioSalazare/multi-copula/tree/master/examples/images/writer_test_profiles.gif)

More examples can be found in the [examples](examples) folder (under development).

## Reading and citations:

The mathematical formulation of the generative model with the copula can be found at:

> *"Conditional Multivariate Elliptical Copulas to Model Residential Load Profiles From Smart Meter Data,"*
E.M. (Mauricio) Salazar Duque, P.P. Vergara, P.H. Nguyen, A. van der Molen and J. G. Slootweg,
in IEEE Transactions on Smart Grid, vol. 12, no. 5, pp. 4280-4294, Sept. 2021, doi: [10.1109/TSG.2021.3078394](https://ieeexplore.ieee.org/document/9425537)


How to contact us
-----------------
Any questions, suggestions or collaborations contact Mauricio Salazar at <e.m.salazar.duque@tue.nl>