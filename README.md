[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MauricioSalazare/multi-copula/master?urlpath=lab/tree/examples)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/MauricioSalazare/multi-copula/blob/master/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multicopula)
![PyPI - Downloads](https://img.shields.io/pypi/dm/multicopula)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/MauricioSalazare/multi-copula)
![PyPI - Version](https://img.shields.io/pypi/v/multicopula)

# MultiCopula
## What is MultiCopula?
 
It is a multivariate probabilistic modelling package, which uses copula theory.

## How to install
The package can be installed via pip using:

```shell
    pip install multicopula
```

## Example:

Run a basic example as:

```python
from multicopula import EllipticalCopula
import numpy as np


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

```

## Electricity consumption application   
The package can be used to simulate daily electrical consumption profiles (active and reactive power) for 
low voltage and medium voltage networks. 

In this application, the daily load profile (15-minute resolution) is modeled with a copula where each time step 
is a stochastic variable. i.e., $`x_1, \ldots, x_{96}`$. Additionally, the model has an extra variable representing
the annual energy consumption ($`w`$) in GWh. Therefore, the copula model represents a probability distribution of 
the form $`f(x_1, \ldots, x_{96}, w)`$.

The idea is to generate consistent daily profiles depending on the increase in annual energy consumption. In other
words, to create profiles conditioned to the variable $`w`$ to a specific value of annual energy consumption
$`\hat{w}`$. That means a conditioned copula model of the form $`f(x_1, \ldots, x_{96}| w=\hat{w})`$.
The following simulation is an example of generated profiles conditioned to different annual energy consumption values.
The annual values are highlighted with a $`\color{cyan}{\textsf{lorem ipsum}}`$ line in the subplot colorbar. 
The three rows of subplots show different types of electricity consumption.

<p align="center">
<img src="https://github.com/MauricioSalazare/multi-copula/blob/master/examples/images/writer_test_profiles.gif?raw=true" width="600" height="700" />
</p>

More examples can be found in the [examples](examples) folder (under development).

## Reading and citations:

The mathematical formulation of the generative model with the copula can be found at:

> *"Conditional Multivariate Elliptical Copulas to Model Residential Load Profiles From Smart Meter Data,"*
E.M. (Mauricio) Salazar Duque, P.P. Vergara, P.H. Nguyen, A. van der Molen and J. G. Slootweg,
in IEEE Transactions on Smart Grid, vol. 12, no. 5, pp. 4280-4294, Sept. 2021, doi: [10.1109/TSG.2021.3078394](https://ieeexplore.ieee.org/document/9425537)


How to contact us
-----------------
Any questions, suggestions or collaborations contact Mauricio Salazar at <e.m.salazar.duque@tue.nl>