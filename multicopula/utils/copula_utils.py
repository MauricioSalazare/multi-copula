import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, chi2, norm, t
from scipy.special import gamma
from scipy import optimize, interpolate
from scipy.interpolate import griddata
import time
import seaborn as sns
import scipy.stats as sps
import matplotlib.ticker as ticker
from datetime import timedelta


def elli_distribution(data, mean, dim, covariance, nu=None, dist='gaussian'):
    """
    Calculate the values of the samples (data) on the probability density function (p.d.f) for a 'gaussian' or
    't-student' distributions.

    The method broadcast the function over the data samples. This makes the calculation a lot of faster for large
    dataset samples, making easier and faster the calculations of the log-likelihood.

    The elliptical distribution function follows the notation of Claudia Czado [1]

        f(x; mu, Sigma) = k_d |Sigma|^(-1/2) * g( (x-mu)^T · Sigma · (x-mu) )

        Sigma: Covariance matrix
        mu: Mean vector
        T: Transpose marker

        Where k_d and g(t) are defined as:

        't-student':
            k_d = gamma( (nu + d) / 2)  / ( gamma(nu/2) * (nu * d) ^ (d/2) )

            g(t) =  ( 1 + t/nu )^(-(v + d)/2)

            nu: Degrees of freedom
            d: dimensions (number of variables)
            gamma: gamma distribution (generalization of n!)

        'Gaussian':
            k_d = (2 * pi)^(-d/2)

            g(t) = exp(-t / 2)

            d: dimensions (number of variables)

    [1] - Czado, Claudia. "Analyzing Dependent Data with Vine Copulas." Lecture Notes in Statistics, Springer (2019).
          pages 4 - 8.

    Input:
    -----
        data: (obj::numpy array): 2D - Array with dimension [dim x N]: 'N' are number of samples an
                                  'dim' are the number of variables.
                                  3D - Array with dimension [N x N x dim]: This is used for broadcasting a combination
                                  of variables using the mesh function.
        mean: (obj::numpy array): 2D - Array with dimensions [dim x 1]: 'dim' are number of variables
        dim: (int):  The number of dimension/variables. This is for sanity check that the user knows
                      how many dimension the problem has.
        covariance: (obj:: numpy array):  2D- Array with dimensions [dim x dim]
        nu: (int):  Degrees of Freedom for the multivariate t-student distribution
        dist: (str):  The dist of distribution to be calculated. Only 2 options available:
                      'gaussian' or 't'.

    Return:
    ------:
        (obj:: numpy array):  1D - Vector with dimension [N,] with the values of the samples evaluated in
                              the p.d.f. selected.
    """
    assert (mean.shape == (dim, 1)),  "Mean matrix has incorrect dimensions"
    assert (len(data.shape) < 4), "Data/Samples Matrix needs to have maximum 3-dimensions"
    assert (dist == 'gaussian' or dist == 't'), "Select the correct type of distribution"

    if len(data.shape) == 2: # The array is 2D
        x_m = data.reshape(dim, -1) - mean.reshape(dim, 1)
    else:
        x_m = data.reshape(-1, dim).T - mean.reshape(dim, 1)

    t_ = np.sum(x_m * np.linalg.solve(covariance, x_m), axis=0)

    g_t_ = g_t(t_, dim=dim, nu=nu, dist=dist)
    k_d_ = k_d(dim=dim, nu=nu, dist=dist)

    #TODO: If the determinant of the covariance is 0, everything is doomed == singular matrix
    pdf = k_d_ * 1 / np.sqrt(np.linalg.det(covariance)) * g_t_

    # determinant = np.linalg.det(covariance)
    #
    # if determinant == 0.0:
    #     determinant = -10 ** -200
    #
    # pdf = k_d_ * (1 / np.sqrt(determinant)) * g_t_

    if len(data.shape) == 2:  # The array is 2D
        return pdf
    else:  # The array is 3D
        return pdf.reshape(data.shape[:-1])


def g_t(x, dim=None, nu=None, dist='gaussian'):
    if dist == 'gaussian':
        return np.exp(- x / 2)
    elif dist == 't':
        assert (dim >= 2),  "The dimension should be at least a bivariate problem"
        assert (dim is not None),  "No scalar in the dimension variable"
        assert (nu is not None),  "No scalar in 'nu' (degrees of freedom - DoF)"
        # assert nu >= 2  # Approximation works for a DoF greater than 2
        return np.power(1 + x / nu, -(nu + dim) / 2)
    else:
        raise ValueError('Wrong distribution selected')


def k_d(dim=None, nu=None, dist='gaussian'):
    assert (dim >= 2),  "The dimension should be at least a bivariate problem"
    assert (dim is not None),  "No scalar in the dimension variable"

    if dist == 'gaussian':
        return np.power(2 * np.pi, -dim / 2)
    elif dist == 't':
        assert (nu is not None),  "You need nu (degrees of freedom - DoF)"
        # assert (nu >= 2),  "Approximation works for a DoF greater than 2"
        return gamma((nu + dim) / 2) / (gamma(nu / 2) * np.power(nu * np.pi, dim / 2))
    else:
        raise ValueError('Wrong distribution selected')


def is_pos_def(A):
    """
    Check if the matrix A is positive definite:
    https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def samples_multivariate_t(mean, covariance, nu, n_samples, allow_singular=False):
    """
    Multivariate t-Student (MVT) Generator.

    [1] - "On Sampling from the Multivariate t Distribution" - Marius Hofert. The R Journal Vol. 5/2, December 2013.
    ISSN 2073-4859. Page 131. Equation (3)

        X = \mu + sqrt(W) * A * Z

        X: Random samples from a multivariate t-student distribution.
        \mu: Mean of the probability distribution
        W: nu / Chi-squared (nu > 0, Chi-squared distribution)
        A: Cholesky decomposition (lower triangular) of the scale matrix \sigma for a multivariate gaussian.
        Z: Multivariate random gaussian with covariance/scale matrix the identity matrix.

        In python we can say that Y = A * Z. And use the scipy function multivariate normal to do the sampling.
    """

    dim = covariance.shape[0]
    assert (mean.shape == (dim, 1)), "Shape should have dimension 2D dimension with size [dim, 1]"
    # Sanity check, as the optimization should only have solutions for nu > 2, to have a defined covariance.
    assert (nu >= 2),  "The approximation only works for  ' v (DoF) > 2' "

    q = chi2(df=nu).rvs(n_samples).reshape(-1, 1) / nu
    y = multivariate_normal(np.zeros(len(covariance)),
                            covariance,
                            allow_singular=allow_singular).rvs(n_samples)

    return np.divide(y, np.sqrt(q)).transpose() + mean


def plot_samples(data_samples):
    """
    Plot data_samples for 1, 2, or 3 variables. If data_samples has more than 3 variables, don't use this method.
    """

    assert (len(data_samples.shape) == 2),  "Array should be 2-D"
    ax = None

    if data_samples.shape[0] == 1:  # Univariate
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.hist(data_samples.ravel(), bins=100, histtype='step')
        plt.show()

    elif data_samples.shape[0] == 2:  # Bivariate case
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(data_samples[0, :], data_samples[1, :], marker='.', s=5)
        ax.set_title('Data samples')
        plt.show()

    elif data_samples.shape[0] == 3:  # Trivariate case
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(data_samples[0, :], data_samples[1, :],  data_samples[2, :], marker='.', s=5)
        ax.set_title('Data samples')
        plt.show()

    return ax


def conditional_parameters(dim, mean_vector, covariance_kendall, nu=None, copula_type='gaussian',  variables={'x2':3}):
    r"""
    Calculate the conditional parameters: covariance (\sigma), mean (\mu) and degrees of freedom (\nu),
    for the elliptical distributions. The notation is the following:

        Covariance block matrix:
        -----------------------
            \sigma = [[\sigma_{aa}    ,  \sigma_{ab}],
                      [\sigma_{ab}^{T},  \sigma_{bb}]]

            \sigma{ba} == \sigma{ab}^{T}

        Conditional mean:
        -----------------
            \mu{a|b} = \mu_{a} + \sigma_{ab}^{T} * \sigma_{bb}^{-1} * (x_{b} - \mu_{b})

        Conditional covariance:
        -----------------------
            \sigma_{a|b} = k_cond * \sigma_{aa} - \sigma_{ab}^{T} * \sigma_{bb}^{-1} * \sigma_{ba}

            k_cond = 1   for  'gaussian'
            k_cond = (\nu + (x_{b} - \mu_{b})^{T} * \sigma_{bb}^{-1} * (x_{b} - \mu_{b})) / (\nu + d_{b})

            where d_{b}: Dimension of the known variables (e.g. how many variables are conditioned)

        Conditional degrees of freedom (nu):
        ------------------------------------
            \nu_{a|b} = \nu + d_{b}


    Return:
    ------
        mu_cond: (obj:: numpy.array)
                2-D numpy array with dimension [(D - P) x 1]. P: Dimension of known variables.
                (e.g. variables={'x2': 3.5, 'x4': 6.9}, then P = 2)

        sigma_cond:
        (obj:: numpy.array)
                2-D numpy array with dimension [(D - P) x (D - P)]

        nu_cond:
        (obj:: numpy.array)
                2-D numpy array with dimension [1 x 1]

    """
    assert ((len(mean_vector.shape) == 2) and (len(covariance_kendall.shape) == 2)), "Mean and covariance should be 2-D"
    assert (mean_vector.shape[0] == covariance_kendall.shape[0]), "Mean and covariance has wrong dimensions"
    assert (copula_type.lower() in ['gaussian', 't']), "Wrong copula type selected"

    known_var_idx = []
    value_var = []
    for key in variables.keys():
        value_var.append(float(variables[key]))
        known_var_idx.append(int(key.replace('x', '')) - 1)
    known_var_idx = np.array(known_var_idx)
    value_var = np.array(value_var)

    assert ((dim - known_var_idx.max()) > 0), 'Cond. variables has higher or equal dimension than model'
    assert ((dim - len(known_var_idx)) > 0), 'Number of cond. variables are more than dimensions in the model'

    shift_idx = np.array([False] * dim)
    shift_idx[known_var_idx.tolist()] = True

    # variables_num = np.linspace(0, dim - 1, dim, dtype=np.int16)
    # variables_num = variables_num[shift_idx]

    # for ii, value in enumerate(variables_num):
    #     value_var[ii] = self.ecdf[value](value_var[ii])  # Transform the variable value to uniform hyper cube
    #
    #     if copula_type == 'gaussian':
    #         value_var[ii] = norm.ppf(value_var[ii])  # Transform to the normal space (\phi^{-1})
    #     else:  # 't' copula
    #         value_var[ii] = t(df=nu).ppf(value_var[ii])

    value_var = np.array(value_var).reshape(len(value_var), 1)

    # Calculate the conditional covariance, mean and degrees of freedom
    # Pre-locate memory:
    dim_new = dim - len(known_var_idx)
    sigma_cond = np.zeros((dim_new, dim_new))
    mu_cond = np.zeros((dim_new, 1))
    d_B = len(known_var_idx)  # Dimensions of the known variables d_{b}

    # --------------------------------------
    # SIGMA CONDITIONAL:  Sigma_(a|b)
    # --------------------------------------
    # Block A will be the one to marginalize. p(x_a | x_b).
    # Meaning: a -> unknowns   b -> known, provided, fixed values
    # Covariance matrix will be build as:
    # | A   B |
    # | B^T D |

    cov_matrix = np.array(covariance_kendall)

    sigma_D = cov_matrix[shift_idx, :][:, shift_idx]
    sigma_A = cov_matrix[~shift_idx, :][:, ~shift_idx]
    sigma_B = cov_matrix[~shift_idx, :][:, shift_idx]

    # --------------------------------------
    # MEAN CONDITIONAL:  Mu_(a|b)
    # --------------------------------------
    # Means organized to follow the same convention
    # | mu_a |
    # | mu_b |

    # mean_vector = np.array(np.zeros((dim, 1)))

    mu_A = mean_vector[~shift_idx]
    mu_B = mean_vector[shift_idx]

    if copula_type == 'gaussian':
        k_cond = 1
    else:
        k_cond = ((nu + np.matmul(np.matmul((value_var - mu_B).T, np.linalg.inv(sigma_D)), (value_var - mu_B)))
                  / (nu + d_B))

    sigma_cond[:, :] = k_cond * (sigma_A - np.matmul(np.matmul(sigma_B, np.linalg.inv(sigma_D)), sigma_B.T))
    mu_cond[:] = mu_A + np.matmul(np.matmul(sigma_B, np.linalg.inv(sigma_D)), (value_var - mu_B))

    if copula_type == 't':
        # --------------------------------------
        # NU (Degrees of Freedom - DoF) CONDITIONAL:  Nu_(a|b)
        # --------------------------------------
        # DoF organized to follow the same convention
        # | nu_a |
        # | nu_b |

        nu_cond = nu + d_B

    else:
        nu_cond = None

    unknown_variables_index = ~shift_idx

    return mu_cond, sigma_cond, nu_cond, unknown_variables_index


def covariance_kendall_tau(data_samples):
    # assert (data_samples.shape[1] > data_samples.shape[0]), "Samples should be greater than number of variables"  # TODO: The original file has this uncommented

    tau = pd.DataFrame(data_samples).T.corr(method='kendall').values
    spearman_rho = pd.DataFrame(data_samples).T.corr(method='spearman').values

    return (np.sin((np.pi * tau) / 2),  # Pearson relation with kendall's tau
            tau,   # Kendall's tau matrix
            2 * np.sin((np.pi / 6) * spearman_rho),  # Pearson relation with spearman's rho
            spearman_rho)  # Spearman rho matrix


def neg_log_likelihood_t_plot(data_samples, mean, covariance, dim, upper_bnd=100, step_size=300):
    start = time.time()
    log_likelihood = []
    nu_range = np.linspace(2, upper_bnd, step_size)
    for nu__ in nu_range:
        ans_t = elli_distribution(data=data_samples, mean=mean, dim=dim,
                                  covariance=covariance, nu=nu__, dist='t')
        log_likelihood.append(np.sum(-np.log(ans_t)))

    log_likelihood = np.array(log_likelihood)
    best_nu = nu_range[np.argmin(log_likelihood)]

    print(f'Best nu value: {best_nu}')
    print(f'Time processing: {time.time() - start}')

    ans_t = elli_distribution(data=data_samples, mean=mean, dim=dim, covariance=covariance,
                              nu=best_nu, dist='t')
    # idx = (ans_t == np.inf)  # Clean the values that generates and error
    print(f'Value of the log-likelihood: {np.sum(-np.log(ans_t))}')

    plt.figure()
    plt.plot(nu_range, log_likelihood)
    plt.title('negative log-likelihood "t-student"')
    plt.xlabel('nu - (degrees of freedom - DoF)')
    plt.ylabel('Neg-Log-likelihood')
    plt.show()


def neg_log_likelihood_t(x, *params):
    """
    Wrapper function over the elliptical distribution function to calculate the negative log-likelihood of the data,
    with a parameter 'nu' (Degrees of Freedom)
    """
    values = -np.log(elli_distribution(data=params[0],
                                       mean=params[1],
                                       dim=params[2],
                                       covariance=params[3],
                                       nu=x,
                                       dist=params[4]))
    # idx = (values == np.inf)  # Clean the values that generates and error

    return np.sum(values)


def optimize_nu(samples, mean, covariance, dim, plot=True):
    n = np.floor(samples.shape[1] * 0.8).astype(np.int)
    nu_bounds = ((0, 200),)
    nu_results = []
    for _ in range(200):
        columns = np.random.randint(samples.shape[1], size=n)
        result = optimize.minimize(neg_log_likelihood_t,
                                   x0=np.array(3),
                                   method='SLSQP',
                                   bounds=nu_bounds,
                                   args=(samples[:, columns],
                                         mean,
                                         dim,
                                         covariance,
                                         't'))
        nu_results.append(result.x)
    nu_results = np.array(nu_results).squeeze()
    low_quantile = np.quantile(nu_results, 0.025)
    high_quantile = np.quantile(nu_results, 0.975)

    if plot:
        plt.figure()
        plt.hist(nu_results)
        plt.title('Optimal nu results - Histogram')
        plt.xlabel('nu - Degrees of Freedom (DoF)')
        plt.show()

    print('-------------------------')
    print('Stochastic "nu"  results:')
    print('-------------------------')
    print(f'nu mean: {nu_results.mean().round(3)}')
    print(f'nu low quantile (2.5%): {low_quantile.round(3)}')
    print(f'nu high quantile (97.5%): {high_quantile.round(3)}')

    return nu_results.mean().round(3), low_quantile.round(3), high_quantile.round(3)

#
# def pit(X):
#     """
#     Takes a data array X of dimension [M x N], and converts it to a uniform
#     random variable using the probability integral transform, U = F(X)
#     """
#     M = X.shape[0]
#     N = X.shape[1]
#
#     # convert X to U by using the probability integral transform:  F(X) = U
#     U = np.empty(X.shape)
#     for ii in range(0, N):
#         x_ii = X[:, ii]
#
#         # estimate the empirical cdf
#         (xx, pp) = ecdf(x_ii, M)
#         f = interpolate.interp1d(xx, pp)
#
#         # plug this RV sample into the empirical cdf to get uniform RV
#         u_ii = f(x_ii)
#         U[:, ii] = u_ii
#
#     return U


def ecdf(x_i, npoints):
    """ Generates an Empirical CDF using the indicator function.

    Inputs:
    x_i -- the input data set, should be a numpy array
    npoints -- the number of desired points in the empirical CDF estimate

    Outputs:
    y -- the empirical CDF
    """
    # define the points over which we will generate the kernel density estimate
    x = np.linspace(min(x_i), max(x_i), npoints)
    n = float(x_i.size)
    y = np.zeros(npoints)

    for ii in np.arange(x.size):
        idxs = np.where(x_i <= x[ii])
        y[ii] = np.sum(idxs[0].size) / (n + 1)

    return (x, y)


def probability_integral_transform(data, plot=False, variable=1, interpolation='spline', bins=None):
    '''
    Transforms the data to the uniform space, using and empirical distribution function.
    The method also returns a spline model of the ECDF and inverser of ECDF for future data sets.

    The empirical distribution function is take from [1]:

            \hat{F}(x) = 1/(n + 1) \sum_{n}{i=1}  1{x_i <= x}  for all x

        Where
        1: The indicator function.
        n: Number of samples.
        'n + 1' is used instead of 'n' to avoid boundary problems of the estimator \hat{F}(x).

    [1] - Czado, Claudia. "Analyzing Dependent Data with Vine Copulas." Lecture Notes in Statistics, Springer (2019).
          page 3.


    The output is the linear interpolation between \hat{F}(x) and \hat{x}, which \hat{x} are values equally
    spaced between the minimum and the maximum of 'x'.

    Notes on interpolation:
    The spline interpolation in scipy fails if you have repeated values over the x-axis, it should have only
    unique values, which is not the case for real data. Therefore, a np.linspace should be done to create an array
    that represents the values in the x-axis for the interpolation.

    The most difficult parts to interpolate are around 0 and 1, If the conditional copula is on the limits,
    you can se artifacts in the simulated data, because of the interpolation.

    Input:
    ------
        data (obj:: numpy array):  The rows are variables and columns instances of the variables.
        plot (bool):    Plots for the visual inspection of the transformation.

    Returns:
    --------
        uniform_samples (obj:: numpy array):  Values within [0,1], which is the transformation of the input
                                              data into the uniform space.
        ecdf_model (obj:: scipy.interp1d):  Model with the spline of the ecdf
        inv_ecdf_model (obj:: scipy.interp1d):  Model with the splint of the inverse of the ecdf.

    '''
    #%%
    ecdf_models = []
    inv_ecdf_models = []
    uniform_values = []

    for ii in range(data.shape[0]):
        '''ECDF Calculation per variable'''

        x_data = data[ii, :]
        n_obs = data[ii, :].shape[0]
        _x = np.linspace(data[ii, :].min(), data[ii, :].max(), n_obs)
        _y = np.empty(n_obs)

        # Avoid boundary problems in the spline and linear model
        for jj in np.arange(n_obs):
            _y[jj] = np.sum(x_data <= _x[jj]) / (n_obs + 1)

        # Avoid boundary problems in the linear model
        _x_bnd = np.r_[-np.inf, _x, np.inf]
        _y_bnd = np.r_[0.0, _y, 1.0]

        if interpolation == 'linear':
            ecdf_fun = interpolate.interp1d(_x_bnd, _y_bnd)
            inv_ecdf = interpolate.interp1d(_y_bnd, _x_bnd)

            ecdf_models.append(ecdf_fun)
            inv_ecdf_models.append(inv_ecdf)
            uniform_values.append(ecdf_fun(data[ii, :]))
        else:
            ecdf_fun_tck = interpolate.splrep(_x, _y)
            inv_ecdf_tck = interpolate.splrep(_y, _x)

            ecdf_models.append(ecdf_fun_tck)
            inv_ecdf_models.append(inv_ecdf_tck)
            uniform_values.append(interpolate.splev(data[ii, :], ecdf_fun_tck))

    uniform_values = np.array(uniform_values)

    if plot:
        fig = plt.figure(figsize=(15, 4))
        ax = fig.subplots(1, 4)

        if interpolation == 'linear':
            ecdf_x_support = ecdf_models[variable].x
            ecdf_y_support = ecdf_models[variable].y

            inv_ecdf_x_support = inv_ecdf_models[variable].x
            inv_ecdf_y_support = inv_ecdf_models[variable].y

            uniform_transform = ecdf_models[variable](data[variable, :])
        else:
            ecdf_x_support = ecdf_models[variable][0]
            ecdf_y_support = interpolate.splev(ecdf_models[variable][0], ecdf_models[variable])

            inv_ecdf_x_support = inv_ecdf_models[variable][0]
            inv_ecdf_y_support = interpolate.splev(inv_ecdf_models[variable][0], inv_ecdf_models[variable])

            uniform_transform =  interpolate.splev(data[variable, :], ecdf_models[variable])


        ax[0].hist(data[variable, :], bins=bins, histtype='step', label=variable)
        ax[0].legend()

        ax[1].plot(ecdf_x_support, ecdf_y_support, lw=0.5, label='CDF')
        ax[1].legend()

        ax[2].plot(inv_ecdf_x_support,inv_ecdf_y_support, lw=0.5, label='Inverse CDF')
        ax[2].legend()

        ax[3].hist(uniform_transform, bins=bins, histtype='step',
                   label= 'Uniform dist. (Transformed)')
        ax[3].legend(loc='lower center')
        plt.suptitle('Probability Integral Transform (PIT) - Variable: ' + str(variable)
                     + '\nInterpolation method: ' + interpolation)
        plt.show()

    return uniform_values, ecdf_models, inv_ecdf_models

def t_copula(uniform_values, covariance, nu, dim):
    """
    't-student' copula density
    """

    t_student = t(df=nu)
    c_density = elli_distribution(data=t_student.ppf(uniform_values), mean=np.zeros((dim, 1)),
                                  dim=dim, covariance=covariance, nu=nu, dist='t')

    if len(uniform_values.shape) == 2:  # 2-D Matrix
        c_normalize = np.ones((1, uniform_values.shape[1]))
        for ii in range(dim):
            c_normalize = c_normalize * t_student.pdf(t_student.ppf(uniform_values[ii, :]))

        #TODO: Remove the division by 0
        # c_normalize[c_normalize == 0.0] = -10**-100
        c_normalize[c_normalize == 0.0] = 10**-100

        c_copula = c_density / c_normalize

    else:  # 3-D Matrix (Used to broadcast the data created by mesh-grid)
        c_normalize = np.ones(uniform_values.shape[0:2])
        for ii in range(dim):
            c_normalize = c_normalize * t_student.pdf(t_student.ppf(uniform_values[:, :, ii]))

        #TODO: Remove the division by 0
        # c_normalize[c_normalize == 0.0] = -10**-100
        c_normalize[c_normalize == 0.0] = 10**-100

        c_copula = c_density / c_normalize

    # print('t copula:')
    # print(f'Nan values: {np.sum(np.isnan(c_copula))}')
    # print(f'inf values: {np.sum(c_copula == np.inf)}')

    return c_copula


def gaussian_copula(uniform_values, covariance, dim):
    """
    Gaussian copula density
    """

    gaussian = norm(loc=0, scale=1)
    c_density = elli_distribution(data=gaussian.ppf(uniform_values), mean=np.zeros((dim, 1)),
                                  dim=dim, covariance=covariance, dist='gaussian')


    if len(uniform_values.shape) == 2:  # 2-D Matrix
        c_normalize = np.ones((1, uniform_values.shape[1]))
        for ii in range(dim):
            c_normalize = c_normalize * gaussian.pdf(gaussian.ppf(uniform_values[ii, :]))

        #TODO: Remove the division by 0
        # c_normalize[c_normalize == 0.0] = -10**-100
        c_normalize[c_normalize == 0.0] = 10**-100

        c_copula = c_density / c_normalize

    else:  # 3-D Matrix (Used to broadcast the data created by mesh-grid)
        c_normalize = np.ones(uniform_values.shape[0:2])
        for ii in range(dim):
            c_normalize = c_normalize * gaussian.pdf(gaussian.ppf(uniform_values[:, :, ii]))

        #TODO: Remove the division by 0
        # c_normalize[c_normalize == 0.0] = -10**-100
        c_normalize[c_normalize == 0.0] = 10**-100
        c_copula = c_density / c_normalize

    return c_copula


def neg_log_likelihood_copula_t(x, *params):
    """
    Wrapper function over the 't-student' copula function to calculate the negative log-likelihood of the data,
    with a parameter 'nu' (Degrees of Freedom)
    """
    values = t_copula(uniform_values=params[0],
                      covariance=params[1],
                      nu=x,
                      dim=params[2])

    #TODO: Remove the negative or 0 values
    values[values <= 0.0] = 10**-100

    values = -np.log(values)

    return np.nansum(values)


def neg_log_likelihood_copula_t_plot(data_samples, covariance, dim, upper_bnd=100, step_size=300, ax=None,
                                     legend_on=True, return_values=False):
    nu_range = np.linspace(2, upper_bnd, step_size)
    log_likelihood = []
    for nu__ in nu_range:
        values = t_copula(uniform_values=data_samples,
                          covariance=covariance,
                          nu=nu__,
                          dim=dim)
        values = -np.log(values)

        log_likelihood.append(np.nansum(values))

    log_likelihood = np.array(log_likelihood)

    log_like_clean = log_likelihood.copy()
    log_like_clean[(log_like_clean == -np.inf)] = np.inf  #  Remove 0.0 values of the evaluation of copula
    best_nu = nu_range[np.argmin(log_like_clean)]

    best_likelihood = t_copula(uniform_values=data_samples,
                               covariance=covariance,
                               nu=best_nu,
                               dim=dim)

    best_likelihood = -np.log(best_likelihood)
    t_neg_loglike = np.nansum(best_likelihood)

    print('\n')
    print('-------------------------------------------')
    print('"t-student" Copula (Linear search)')
    print('-------------------------------------------')
    print(f'Best nu value: {best_nu}')
    print(f'Neg log-likelihood: {t_neg_loglike}')

    values = gaussian_copula(uniform_values=data_samples, covariance=covariance, dim=dim)
    values = -np.log(values)

    gauss_neg_loglike = np.nansum(values)

    print('\n')
    print('-------------------------------------------')
    print('Gaussian Copula')
    print('-------------------------------------------')
    print(f'Neg log-likelihood: {gauss_neg_loglike}')
    print('\n')

    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1,1,1)
    ax.plot(nu_range, log_likelihood, label='t-Copula')
    ax.scatter(best_nu, t_neg_loglike, s=40, facecolors='none', edgecolors='r', label=r'Optimal $\nu$')
    ax.axhline(gauss_neg_loglike, linewidth=0.5, color='k', label='Gaussian-Copula')
    if legend_on:
        ax.legend()
    plt.show()

    if return_values:
        return (nu_range, log_likelihood, best_nu, t_neg_loglike, gauss_neg_loglike)
    return ax


def initial_guess(data):
    nu = []
    for ii in range(data.shape[0]):
        nu_, _, _ = t.fit(data[ii, :])
        nu.append(nu_)

    return np.array(nu).mean()


def normalize_copulas_visualization():
    """
    Method to show the 't-student' and 'gaussian' copula in the normalized versions as visual aid
    """
    tau = 0.2
    rho = np.sin(tau * np.pi / 2)
    scale = [[1, rho],
             [rho, 1]]
    nu = 4

    xx, yy = np.meshgrid(
        np.linspace(-8, 8, 500),
        np.linspace(-8, 8, 500))

    # xx_ = norm.cdf(norm.ppf(xx))
    uniform_z_x = t(df=nu).cdf(xx)
    uniform_z_y = t(df=nu).cdf(yy)

    z_x = norm.ppf(uniform_z_x)
    z_y = norm.ppf(uniform_z_y)
    pos = np.dstack((z_x, z_y))  # This is Z

    values = t_copula(uniform_values=norm.cdf(pos), covariance=np.array(scale), nu=nu, dim=2)

    rr_1 = norm.pdf(pos[:, :, 0])
    rr_2 = norm.pdf(pos[:, :, 1])

    re_values = values * rr_1 * rr_2

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    cs = ax.contour(z_x, z_y, re_values, 10, linewidths=0.8)
    ax.clabel(cs, inline=1, fontsize=8)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    values_gauss = gaussian_copula(uniform_values=norm.cdf(pos), covariance=np.array(scale), dim=2)
    re_values = values_gauss * rr_1 * rr_2
    ax = fig.add_subplot(122)
    cs = ax.contour(z_x, z_y, re_values, 10, linewidths=0.8)
    ax.clabel(cs, inline=1, fontsize=8)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])




def quarter_converter(quarter):
    hour = timedelta(minutes=(quarter) * 15).seconds // 3600
    minutes = (timedelta(minutes=(quarter) * 15).seconds // 60) % 60

    if minutes == 0:
        minutes_str = '00'
    else:
        minutes_str = str(minutes)

    return str(hour) + ':' + minutes_str


def plot_standarized_samples(samples):
    uniform_samples, _,_ =probability_integral_transform(samples)
    cov_pearson, tau, _, _ = covariance_kendall_tau(samples)
    standarized_plots(uniform_samples, [0,1], pearson=cov_pearson, tau=tau)



def standarized_plots(uniform_samples, variables, pearson, tau, ax=None):

    n_grid = len(variables)

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.subplots(n_grid, n_grid)
        fig.subplots_adjust(wspace=0, hspace=0)

    # Lower diagonal
    for col in range(n_grid - 1):
        for row in range(col + 1, n_grid):
            # var_1 = 60
            # var_2 = 70
            uniform_z_x = uniform_samples[variables[row], :]
            uniform_z_y = uniform_samples[variables[col], :]

            # z-scale of observations
            z_x = norm.ppf(uniform_z_x)
            z_y = norm.ppf(uniform_z_y)
            z_i = np.array([z_x, z_y])

            kde = sps.gaussian_kde(z_i, bw_method=0.5)
            # get a regular grid of points over our region of interest
            xx, yy = np.meshgrid(
                np.linspace(-3, 3, 50),
                np.linspace(-3, 3, 50))
            # calculate probability density on these points
            z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
            cs = ax[row, col].contour(xx, yy, z, 6, linewidths=0.5, cmap=plt.get_cmap('plasma'))
            # ax[row, col].clabel(cs, inline=1, fontsize=4)
            ax[row, col].set_xlim([-3, 3])
            ax[row, col].set_ylim([-3, 3])
            ax[row, col].yaxis.set_major_formatter(ticker.NullFormatter())
            ax[row, col].xaxis.set_major_formatter(ticker.NullFormatter())
            ax[row, col].set_xticks([], [])
            ax[row, col].set_yticks([], [])

    # Upper-diagonal
    for row in range(n_grid - 1):
        for col in range(row + 1, n_grid):
            # ax[row, col].scatter(uniform_samples[row, :], uniform_samples[col, :], s=5, marker='.', c='#CCCCCC')
            ax[row, col].scatter(uniform_samples[row, :], uniform_samples[col, :], s=2, marker='.', c='k')
            # ax[row, col].text(0.5, 0.5, "{:.{}f}".format(tau[row, col], 2),
            #                   horizontalalignment='center',
            #                   verticalalignment='center',
            #                   transform=ax[row, col].transAxes,
            #                   fontdict={'color': 'red', 'weight': 'bold', 'size': 12},
            #                   bbox=dict(facecolor='w', edgecolor='w'))
            # ax[row, col].text(0.5, 0.6, "{:.{}f}".format(pearson[row, col], 2),
            #                   horizontalalignment='center',
            #                   verticalalignment='center',
            #                   transform=ax[row, col].transAxes,
            #                   fontdict={'color': 'blue', 'weight': 'bold', 'size': 12})
            ax[row, col].yaxis.set_major_formatter(ticker.NullFormatter())
            ax[row, col].xaxis.set_major_formatter(ticker.NullFormatter())
            ax[row, col].set_xticks([], [])
            ax[row, col].set_yticks([], [])

    # Diagonal
    for diag in range(n_grid):
        ax[diag, diag].hist(uniform_samples[diag], density=True, edgecolor='w', fc='#AAAAAA')
        ax[diag, diag].set_ylim([0, 1.5])

        if variables[diag] != 96:
            # ax[diag, diag].text(x=0.5, y=0.8, s='quarter.' + str(variables[diag]),
            #                     horizontalalignment='center',
            #                     verticalalignment='center',
            #                     transform=ax[diag, diag].transAxes,
            #                     fontdict={'color': 'red', 'weight': 'bold'})
            ax[diag, diag].text(x=0.5, y=0.8, s=quarter_converter(variables[diag]),
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=ax[diag, diag].transAxes,
                                fontdict={'color': 'red', 'weight': 'bold', 'size': 9})
        else:
            ax[diag, diag].text(x=0.5, y=0.8, s='energy.year',
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=ax[diag, diag].transAxes,
                                fontdict={'color': 'red', 'weight': 'bold', 'size': 7})


        ax[diag, diag].hlines(1.0, xmin=ax[diag, diag].get_xlim()[0], xmax=ax[diag, diag].get_xlim()[1],
                              linestyles={'dashed'}, linewidths=0.8, colors='k')
        ax[diag, diag].yaxis.set_major_formatter(ticker.NullFormatter())
        ax[diag, diag].xaxis.set_major_formatter(ticker.NullFormatter())
        ax[diag, diag].set_xticks([], [])
        ax[diag, diag].set_yticks([], [])

    return ax


def plot_covariance_matrix(covariance, ax=None):
    levels = None

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    cb_ax = ax.contourf(covariance, levels=levels, cmap=plt.cm.get_cmap('PuOr'), vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Time intervals [quarters]')
    cbar = plt.colorbar(cb_ax, ax=ax)
    cbar.ax.set_ylabel('Kendall\'s tau Correlation')
    plt.show()

    return ax


def plot_time_steps(samples, xlim=None, ylim=None):
    if xlim==None:
        xlim_ = (-0.1, 3)
    else:
        xlim_ = xlim

    if ylim == None:
        ylim_ = (-0.1, 3)
    else:
        ylim_ = ylim

    return sns.jointplot(samples[0, :], samples[1, :],
                         xlim=xlim_, ylim=ylim_,
                         s=5, fc='k', ec='k', marker='x').plot_joint(sns.kdeplot,
                                                                     n_levels=30,
                                                                     linewidths=0.5,
                                                                     zorder=1)


def plot_uniform_variables(u_, v_):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.subplots(1, 2)
    sns.kdeplot(u_, v_, ax=ax[0])
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[1].scatter(u_, v_, marker='.', s=10)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    plt.show()



def plot_cdf_2d(samples):
    assert (samples.shape[0] ==  2), 'Samples should be in bivariate only'

    samples_trans = samples.T.copy()
    n_obs = samples_trans.size
    z = []
    for xx, yy in samples_trans:
        z.append(np.sum((samples_trans <= xx) & (samples_trans <= yy)) / (n_obs + 1))
    z = np.array(z)

    bivariate_cdf = np.hstack([samples_trans, np.array(z).reshape(-1, 1)])


    # Interpolate the data
    pts = 100j
    x_min = np.floor(bivariate_cdf[:, 0].min())
    x_max = np.ceil(bivariate_cdf[:, 0].max())
    y_min = np.floor(bivariate_cdf[:, 1].min())
    y_max = np.ceil(bivariate_cdf[:, 1].max())
    X, Y = np.mgrid[x_min:x_max:pts, y_min:y_max:pts]
    F = griddata(bivariate_cdf[:,0:2], bivariate_cdf[:,2], (X, Y))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, F, cmap=plt.get_cmap('viridis'), norm=plt.Normalize(vmax=np.nanmax(F), vmin=np.nanmin(F)))
    ax.set_zlim([0, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('ECDF')
    ax.set_title('Bivariate CDF (Empirical CDF)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return ax



def plot_pit(data, variable, interpolation='linear', bins=None):

    if isinstance(variable, list):
        if len(variable) == 1:
            variable = variable[0]
            probability_integral_transform(data=data,
                                           plot=True,
                                           variable=variable,
                                           interpolation=interpolation,
                                           bins=bins)
        else:
            for variable_number in variable:
                probability_integral_transform(data=data,
                                               plot=True,
                                               variable=variable_number,
                                               interpolation=interpolation,
                                               bins=bins)
    elif isinstance(variable, int):
        probability_integral_transform(data=data,
                                       plot=True,
                                       variable=variable,
                                       interpolation=interpolation,
                                       bins=bins)

    else:
        raise Warning('The variable is not a list or a integer number')
