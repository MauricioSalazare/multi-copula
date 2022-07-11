.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/MauricioSalazare/multicopula/master?urlpath=lab/tree/examples
   :alt: binder



MultiCopula
===============


What is MultiCopula?
------------------------

It is a multivariate probabilistic modelling package, which uses copula theory.

How to install
--------------
The package can be installed via pip using:

.. code:: shell

    pip install multicopula

Example:
--------
Run the load base case as:

.. code-block:: python

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
   samples_ = copula_model.sample(10000)
   covariance_samples = np.corrcoef(samples_)

More examples can be found in the examples folder (under development).

Reading and citations:
----------------------
The mathematical formulation of the power flow can be found at:

"Conditional Multivariate Elliptical Copulas to Model Residential Load Profiles From Smart Meter Data,"
E.M. (Mauricio) Salazar Duque, P.P. Vergara, P.H. Nguyen, A. van der Molen and J. G. Slootweg,
in IEEE Transactions on Smart Grid, vol. 12, no. 5, pp. 4280-4294, Sept. 2021, doi: 10.1109/TSG.2021.3078394.
`link <https://ieeexplore.ieee.org/document/9425537>`_


How to contact us
-----------------
Any questions, suggestions or collaborations contact Mauricio Salazar at <e.m.salazar.duque@tue.nl>