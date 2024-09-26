# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cubsh000-module:

CUB models in Python.
Module for CUBSH (Combination of Uniform
and Binomial with Shelter Effect).

Description:
============
    This module contains methods and classes
    for CUBSH model family.

Manual, Examples and References:
================================
    - `Models manual <manual.html#cubsh-without-covariates>`__

List of TODOs:
==============
  - TODO: fix 3d plots legend
  - TODO: test all ``def _*():`` (optional functions)

Credits
==============
    :Author:      Massimo Pierini
    :Date:        2023-24
    :Credits:     Domenico Piccolo, Rosaria Simone
    :Contacts:    cub@maxpierini.it

Classes and Functions
=====================
"""

#import pickle
import datetime as dt
import numpy as np
#import pandas as pd
from scipy.special import binom
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, probbit, dissimilarity,
    conf_ell, plot_ellipsoid,
    #chisquared,
    InvalidCategoriesError,
    lsat, luni, aic, bic,
)
from . import cub
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def pidelta_to_pi1pi2(pi, delta):
    r"""Compute :math:`(\pi_1, \pi_2)` from :math:`(\pi, \delta)`

    :math:`\pi_1 = (1 - \delta) \pi`

    :math:`\pi_2 = (1 - \delta)(1 - \pi)`

    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: a tuple of :math:`(\pi_1, \pi_2)` the mixing coefficient of the shifted
        Binomial and the Uniform components, respectively
    :rtype: tuple
    """
    pi1 = (1-delta)*pi
    pi2 = (1-delta)*(1-pi)
    return pi1, pi2

def pi1pi2_to_pidelta(pi1, pi2):
    r"""Compute :math:`(\pi, \delta)` from :math:`(\pi_1, \pi_2)`

    :math:`\pi = \dfrac{\pi_1}{\pi_1 + \pi_2}`

    :math:`\delta = 1 - \pi_1 - \pi_2`

    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :return: a tuple of :math:`(\pi, \delta)` the parameters of 
        uncertainty and shelter choice, respectively
    :rtype: tuple
    """
    pi = pi1/(pi1+pi2)
    delta = 1 - pi1 - pi2
    return pi, delta

def _pidelta_to_lambdeta(pi, delta):
    lambd = pi*(1-delta)
    eta = ((1-pi)*(1-delta))/(1-pi*(1-delta))
    return lambd, eta

def _lambdeta_to_pidelta(lambd, eta):
    pi = lambd/(lambd+eta*(1-lambd))
    delta = (1-lambd)*(1-eta)
    return pi, delta

def pmf_delta(m, sh, pi, xi, delta):
    r"""Probability distribution of a specified CUBSH model,
    using canonic parametrization :math:`(\pi, \delta)`.

    :math:`\Pr(R = r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the probability distribution
    :rtype: array
    """
    R = choices(m)
    D = R==sh
    #print(m, pi, xi, R)
    p = delta*D + (1-delta) * ( pi*binom(m-1, R-1) * (1-xi)**(R-1) * xi**(m-R) + (1-pi)/m )
    return p

def pmf(m, sh, pi1, pi2, xi):
    r"""Probability distribution of a specified CUBSH model,
    using alternative parametrization :math:`(\pi_1, \pi_2)`.

    :math:`\Pr(R = r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the probability distribution
    :rtype: array
    """
    R = choices(m)
    D = (R==sh).astype(int)
    p = pi1*probbit(m, xi) + pi2/m + (1-pi1-pi2)*D
    return p

def prob(m, sh, pi1, pi2, xi, r):
    r"""Probability :math:`\Pr(R = r | \pmb\theta)` of a CUBSH model without covariates,
    using alternative parametrization :math:`(\pi_1, \pi_2)`.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param r: ordinal response
    :type r: int
    :return: the probability :math:`\Pr(R = r | \pmb\theta)`
    :rtype: float
    """
    p = pmf(m, sh, pi1, pi2, xi)
    return p[r-1]

def proba_delta(m, sh, pi, xi, delta, r):
    r"""Probability :math:`\Pr(R = r | \pmb\theta)` of a CUBSH model without covariates,
    using canonic parametrization :math:`(\pi, \delta)`.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param r: ordinal response
    :type r: int
    :return: the probability :math:`\Pr(R = r | \pmb\theta)`
    :rtype: float
    """
    #print(m, pi, xi, R)
    D = r==sh
    p = delta*D + (1-delta) * ( pi*binom(m-1, r-1) * (1-xi)**(r-1) * xi**(m-r) + (1-pi)/m )
    #print(p)
    return p

def cmf(m, sh, pi1, pi2, xi):
    r"""Cumulative probability of a specified CUBSH model,
    using alternative parametrization :math:`(\pi_1, \pi_2)`.

    :math:`\Pr(R \leq r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the cumulative probability distribution
    :rtype: array
    """
    return pmf(m, sh, pi1, pi2, xi).cumsum()

def cmf_delta(m, sh, pi, xi, delta):
    r"""Cumulative probability of a specified CUBSH model,
    using canonic parametrization :math:`(\pi, \delta)`.

    :math:`\Pr(R \leq r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the cumulative probability distribution
    :rtype: array
    """
    return pmf_delta(m, sh, pi, xi, delta).cumsum()

def mean_delta(m, sh, pi, xi, delta):
    r"""Expected value of a specified CUBSH model,
    using canonic parametrization :math:`(\pi, \delta)`.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the expected value of the model
    :rtype: float
    """
    mu = cub.mean(m, pi, xi)
    mi = mu + delta*(mu-sh)
    return mi

def var_delta(m, pi, xi, delta):
    r"""Variance of a specified CUBSH model,
    using canonic parametrization :math:`(\pi, \delta)`.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the variance of the model
    :rtype: float
    """
    v = ((1-delta)**2)*cub.var(m, pi, xi)
    return v

def std_delta(m, pi, xi, delta):
    r"""Standard deviation of a specified CUB model,
    using canonic parametrization :math:`(\pi, \delta)`.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the standard deviation of the model
    :rtype: float
    """
    s = np.sqrt(var_delta(m, pi, xi, delta))
    return s

#TODO: skew
def _skew(pi, xi):
    """
    skewness normalized eta index
    """
    return None #pi*(1/2-xi)

#TODO: test mean_diff
def _mean_diff(m, sh, pi1, pi2, xi):
    R = choices(m)
    S = choices(m)
    mu = 0
    for r in R:
        for s in S:
            mu += abs(r-s)*prob(m,sh,pi1,pi2,xi,r)*prob(m,sh,pi1,pi2,xi,s)
    return mu

#TODO: test median
def _median(m, sh, pi1, pi2, xi):
    R = choices(m)
    cp = cmf(m, sh, pi1, pi2, xi)
    M = R[cp>.5][0]
    if M > R.max():
        M = R.max()
    return M

#TODO: test gini
def _gini(m, sh, pi1, pi2, xi):
    ssum = 0
    for r in choices(m):
        ssum += prob(m, sh, pi1, pi2, xi, r)**2
    return m*(1-ssum)/(m-1)

#TODO: test laakso
def _laakso(m, sh, pi1, pi2, xi):
    g = _gini(m, sh, pi1, pi2, xi)
    return g/(m - (m-1)*g)

def loglik(m, sh, pi1, pi2, xi, f):
    r"""Log-likelihood of a CUB model with shelter effect

    Compute the log-likelihood of a CUB model with a shelter effect
    for the given absolute frequency distribution.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param f: Vector of the absolute frequency distribution
    :type f: array
    :return: the log-likehood value
    :rtype: float
    """
    L = pmf(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi)
    #TODO: check log invalid value from mle
    l = (f*np.log(L)).sum()
    return l

def varcov_pxd(m, sh, pi, xi, de, n):
    r"""Variance-covariance matrix for CUB models with shelter effect,
    using canonic parametrization :math:`(\pi, \delta)`.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param de: shelter choice parameter :math:`\delta`
    :type de: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param n: number of ordinal responses
    :type n: int
    :return: the variance-covariance matrix
    :rtype: numpy ndarray
    """
    bb = probbit(m, xi)
    dd = np.repeat(0, m)
    dd[sh-1] = 1
    cp = cub.pmf(m=m, pi=pi, xi=xi)
    pr = pmf_delta(m, sh, pi, xi, de)
    R = choices(m)
    
    d_dpi = (1-de)*(bb - 1/m)
    d_dde = dd - cp
    d_dxi = pi*(1-de)*binom(m-1,R-1)*(
        (1-xi)**(R-2) * xi**(m-R-1) *
        (xi*(1-m) + m - R)
    )
    i11 = n*(d_dpi**2 / pr).sum()
    i22 = n*(d_dxi**2 / pr).sum()
    i33 = n*(d_dde**2 / pr).sum()
    i12 = n*(d_dpi*d_dxi / pr).sum()
    i21 = i12
    i13 = n*(d_dpi*d_dde / pr).sum()
    i31 = i13
    i23 = n*(d_dde*d_dxi / pr).sum()
    i32 = i23
    
    I = np.array([
        [i11, i12, i13],
        [i21, i22, i23],
        [i31, i32, i33],
    ])
    
    if np.any(np.isnan(I)):
        return None
    if np.linalg.det(I) <= 0:
        return None
    V = np.linalg.inv(I)
    #Vpx = V[:2,:2]
    return V

def varcov(m, sh, pi1, pi2, xi, n):
    r"""Variance-covariance matrix for CUB models with shelter effect,
    using alternative parametrization :math:`(\pi_1, \pi_2)`.

    Compute the variance-covariance matrix of parameter estimates of a CUB model with shelter effect.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param n: number of ordinal responses
    :type n: int
    :return: the variance-covariance matrix
    :rtype: numpy ndarray
    """
    R = choices(m)
    pr = pmf(m, sh, pi1, pi2, xi)
    dd = np.repeat(0, m)
    dd[sh-1] = 1
    bb = probbit(m, xi)

    aaa = bb-dd
    bbb = (1/m)-dd
    c4 = pi1*bb*(m-R-xi*(m-1))/(xi*(1-xi))
    atilde = aaa/pr
    btilde = bbb/pr
    ctilde = c4/pr

    d11 = np.sum(aaa*atilde)
    d22 = np.sum(bbb*btilde)
    dxx = np.sum(c4*ctilde)
    d12 = np.sum(bbb*atilde)
    d1x = np.sum(c4*atilde)
    d2x = np.sum(c4*btilde)

    #TODO: infmat in R style?
    infmat = np.ndarray(shape=(3,3))
    infmat[0,0] = d11
    infmat[1,1] = d22
    infmat[2,2] = dxx
    infmat[0,1] = d12
    infmat[1,0] = d12
    infmat[0,2] = d1x
    infmat[2,0] = d1x
    infmat[1,2] = d2x
    infmat[2,1] = d2x

    varmat = np.ndarray(shape=(3,3))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)/n
    return varmat

def init_theta(f, m, sh):
    r"""Preliminary estimators for CUBSH models.

    Computes preliminary parameter estimates of a CUBSH model without covariates for given ordinal
    responses. These preliminary estimators are used within the package code to start the E-M algorithm.

    :param f: array of the absolute frequencies of given ordinal responses
    :type f: array of int
    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :return: a tuple of :math:`(\pi_1^{(0)}, \pi_2^{(0)}, \xi^{(0)})`
    """
    pi1, xi = cub.init_theta(f, m)
    fc = f[sh-1]/f.sum()
    #print("fc", fc)
    deltamax = (m*fc-1)/(m-1)
    delta = np.max([.01, deltamax])
    pi2 = np.max([.01, 1-delta-pi1])
    return pi1, pi2, xi

def plot_simplex(pi1pi2list, ax=None, fname=None):
    r"""Plot simplex of parameters of a CUBSH model.

    .. note:: see the reference DOI 10.1007/s10260-011-0176-x for details

    .. warning:: this function still needs several fixes

    :param pi1pi2list: list of ``[pi1, pi2]`` parameters
    :type pi1pi2list: list
    :param ax: matplotlib axis
    :param fname: if provided, save the plot to ``fname``, defaults to None
    :param fname: str
    """
    # tick length
    tkl = .02
    # tick step
    steps = .05
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = None
    # simplex axes
    ax.plot([-1, 0], [0, 0], "k")
    ax.plot([-1, -.5], [0, np.sin(np.pi/3)], "k")
    ax.plot([-.5, 0], [np.sin(np.pi/3), 0], "k")

    sin = np.sin(np.pi/3)
    cos = np.cos(np.pi/3)
    for tick in np.arange(0, 1, steps)[1:]:
        ### grid
        # delta grid
        ax.plot(
            [-1+tick/2, -tick/2],
            [tick*sin, tick*sin],
            "k", alpha=.1
        )
        # delta ticks
        ax.plot(
            [-1+tick/2-tkl, -1+tick/2],
            [tick*sin, tick*sin], "k"
        )
        # pi1 grid
        ax.plot(
            [-tick, -tick-(1-tick)*cos],
            [0, (1-tick)*sin],
            "k", alpha=.1
        )
        # pi1 ticks
        ax.plot(
            [-tick+tkl*cos, -tick],
            [-tkl*sin, 0], "k"
        )
        # pi2 grid
        ax.plot(
            [-tick, -(tick)*cos],
            [0, (tick)*sin],
            "k", alpha=.1
        )
        # pi2 ticks
        ax.plot(
            [-tick*cos, -(tick-tkl)*cos],
            [tick*sin, (tick+tkl)*sin], "k"
        )
        ### axes
        # pi1
        ax.text(
            -tick+tkl*cos+tkl/2, -tkl*sin,
            f" {tick:.2f}", ha="center", va="top",
            rotation=300
        )
        # delta
        ax.text(
            -1+tick/2-tkl, tick*sin,
            f"{tick:.2f} ", ha="right", va="center"
        )
        # pi2
        ax.text(
            -(tick-tkl/2)*cos,
            (tick+2*tkl)*sin+tkl/4,
            f" {1-tick:.2f}", ha="left", va="center",
            rotation=60
        )
    ### labels
    ax.text(
        -1+.5/2-5*tkl, .5*sin,
        r"$\delta$", ha="right", va="center",
        fontsize=20
    )
    ax.text(
        -.5, -4.5*tkl,
        r"$\pi_1$", ha="center", va="top",
        fontsize=20
    )
    ax.text(
        -(.5-tkl)*cos+6.5*tkl,
        .5*sin,
        r"$\pi_2$", ha="right", va="center",
        fontsize=20
    )
    # no axes and equal
    plt.axis('off')
    ax.set_aspect('equal', 'box')
    # points
    for i, pi1pi2 in enumerate(pi1pi2list):
        pi1 = pi1pi2[0]
        pi2 = pi1pi2[1]
        _, delta = pi1pi2_to_pidelta(pi1, pi2)
        x = -pi1-delta*cos
        y = delta*sin
        ax.scatter(x, y, facecolor="None",
            edgecolor="k", s=200)
        ax.text(x, y, f"{chr(i+65)}", ha="center", va="center")
    if fig is not None:
        fig.tight_layout()
    if fname:
        fig.savefig(fname)
    return fig, ax

###################################################################
# RANDOM SAMPLE
###################################################################

def draw2(m, sh, pi1, pi2, xi, n,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUBSH model,
    using alternative parametrization :math:`(\pi_1, \pi_2)`.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi1: Mixing coefficient for the shifted Binomial component of the mixture distribution :math:`\pi_1`
    :type pi1: float
    :param pi2: Mixing coefficient for the discrete Uniform component of the mixture distribution :math:`\pi_2`
    :type pi2: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param n: number of ordinal responses
    :type n: int
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    if m<= 4:
        print("ERR: Number of ordered categories should be at least 5")
        raise InvalidCategoriesError(m=m, model="cubsh")
    np.random.seed(seed)
    theoric = pmf(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=theoric
        )
    pi, delta = pi1pi2_to_pidelta(pi1,pi2)
    pars = np.array([pi1, pi2, xi,
        pi, delta])
    par_names = np.array([
        "pi1", "pi2", "xi",
        "*pi", "*delta"
    ])
    p_types = [
        "Uniform",
        "Binomial",
        "Feeling",
        "Uncertainty",
        "Shelter"
    ]
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    sample = CUBsample(
        model="CUBSH",
        rv=rv, m=m, sh=sh,
        pars=pars, par_names=par_names,
        p_types=p_types,
        theoric=theoric, diss=diss,
        df=df, formula=formula
    )
    return sample

def draw(m, sh, pi, xi, delta, n,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUBSH model,
    using canonic parametrization :math:`(\pi, \delta)`.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param n: number of ordinal responses
    :type n: int
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    pi1, pi2 = pidelta_to_pi1pi2(pi, delta)
    sample = draw2(m, sh, pi1, pi2, 
       xi, n, seed=seed,
        df=df, formula=formula)
    return sample

###################################################################
# INFERENCE
###################################################################

def mle(sample, m, sh,
    df, formula, maxiter=500, tol=1e-4,
    ass_pars=None
    ):
    r"""Main function for CUB models with a shelter effect

    Estimate and validate a CUB model with a shelter effect.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param ass_pars: dictionary of hypothesized parameters, defaults to None
    :type ass_pars: dictionary, optional
    :param maxiter: maximum number of iterations allowed for running the optimization algorithm
    :type maxiter: int
    :param tol: fixed error tolerance for final estimates
    :type tol: float
    :return: an instance of ``CUBresCUBSH`` (see the Class for details)
    :rtype: object
    :raise: Exception if :math:`m \leq 4`
    """
    if m<= 4:
        print("ERR: Number of ordered categories should be at least 5")
        raise InvalidCategoriesError(m=m, model="cubsh")

    start = dt.datetime.now()
    R = choices(m)
    f = freq(sample=sample, m=m)
    n = sample.size
    dd = (R==sh).astype(int)
    pi1, pi2, xi = init_theta(f=f, m=m, sh=sh)
    l = loglik(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi, f=f)
    niter = 1
    while niter <= maxiter:
        lold = l
        bb = probbit(m=m, xi=xi)
        tau1 = pi1*bb
        tau2 = pi2/m
        denom = tau1+tau2+(1-pi1-pi2)*dd
        tau1 /= denom
        tau2 /= denom
        #tau3 = 1-tau1-tau2
        numaver = np.sum(R*f*tau1)
        denaver = np.sum(f*tau1)
        averpo = numaver/denaver
        # updated estimates
        pi1 = np.sum(f*tau1)/n
        pi2 = np.sum(f*tau2)/n
        xi = (m-averpo)/(m-1)
        if xi < .001:
            xi = .001
            niter = maxiter-1
        l = loglik(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi, f=f)
        lnew = l
        testll = np.abs(lnew-lold)
        if testll < tol:
            break
        else:
            l = lnew
        niter += 1

    if xi > .999: xi = .99
    if xi < .001: xi = .01
    if pi1 < .001: pi1 = .01

    varmat = varcov(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi, n=n)
    end = dt.datetime.now()
    durata = (end-start).total_seconds()
    pi, delta = pi1pi2_to_pidelta(pi1=pi1, pi2=pi2)

    esdelta = np.sqrt(varmat[0,0]+varmat[1,1]+2*varmat[0,1])
    walddelta = delta/esdelta
    pvaldelta = np.round(2*abs(sps.norm.sf(walddelta)), 20)

    espi = np.sqrt((pi1**2)*varmat[1,1] + (pi2**2)*varmat[0,0] - 2*pi1*pi2*varmat[0,1])/((pi1+pi2)**2)
    waldpi = pi/espi
    pvalpi = np.round(2*abs(sps.norm.sf(waldpi)), 20)

    #trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + 3*np.log(trvarmat/3) - np.log(np.linalg.det(varmat))

    stime = np.array([pi1, pi2, xi])
    errstd = np.sqrt(np.diag(varmat))
    wald = stime/errstd
    pval = np.round(2*abs(sps.norm.sf(wald)), 20)
    
    estimates = np.concatenate((
        [pi1, pi2, xi],
        [pi, xi, delta]
    ))
    est_names = np.array([
        "pi1", "pi2", "xi",
        "pi", "xi", "delta"
    ])
    e_types = np.array([
        "Alternative parametrization",
        None, None,
        "Uncertainty", "Feeling",
        "Shelter effect"
    ])
    stderrs = np.concatenate((
        errstd, [espi], [errstd[-1]],
        [esdelta]
    ))
    wald = np.concatenate((
        wald, [waldpi], [wald[-1]],
        [walddelta]
    ))
    pval = np.concatenate((
        pval, [pvalpi], [pval[-1]],
        [pvaldelta]
    ))

    theoric = pmf(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi)
    diss = dissimilarity(f/n, theoric)
    loglikuni = luni(m=m, n=n)
    #xisb = (m-aver)/(m-1)
    #llsb = cub.loglik(m, 1, xisb, f)
    #TODO: use nonzero in lsat?
    #nonzero = np.nonzero(f)
    logliksat = lsat(f=f, n=n)
    # mean loglikelihood
    muloglik = l/n
    # deviance from saturated model
    dev = 2*(logliksat-l)

    #pearson = (f-n*theorpr)/np.sqrt(n*theorpr)
    #X2 = np.sum(pearson**2)
    #relares = (f/n-theorpr)/theorpr

    #LL2 = 1/(1+np.mean((f/(n*theoric)-1)**2))
    #ll2 = (l-llunif)/(logsat-llunif)
    # FF2 is the overall fraction of correct responses, as predicted by the estimated model
    #FF2 = 1-dissim
    AIC = aic(l=l, p=3)
    BIC = bic(l=l, p=3, n=n)

    return CUBresCUBSH(
        model="CUBSH",
        m=m, sh=sh, n=n,
        niter=niter, maxiter=maxiter,
        tol=tol, theoric=theoric,
        estimates=estimates,
        est_names=est_names,
        e_types=e_types,
        stderrs=stderrs,
        wald=wald, pval=pval,
        loglike=l, loglikuni=loglikuni,
        logliksat=logliksat,
        muloglik=muloglik, dev=dev,
        AIC=AIC, BIC=BIC,
        seconds=durata, time_exe=start,
        sample=sample, f=f, varmat=varmat,
        diss=diss, df=df, formula=formula,
        ass_pars=ass_pars,
    )

class CUBresCUBSH(CUBres):
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """

    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        r"""Plots relative frequencies of observed sample, estimated probability distribution and,
        if provided, probability distribution of a known model.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param kind: choose a barplot (``'bar'`` default) of a scatterplot (``'scatter'``)
        :type kind: str
        :param ax: matplotlib axis, if None a new figure will be created, defaults to None
        :type ax: matplolib ax, optional
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        else:
            fig = None
        
        #pi1 = self.estimates[0]
        #pi2 = self.estimates[1]
        pi = self.estimates[3]
        xi = self.estimates[4]
        delta = self.estimates[5]
        title = "CUBSH model    "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$ , $\delta={delta:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        if self.ass_pars is not None:
            genpi1 = self.ass_pars['pi1']
            genpi2 = self.ass_pars['pi2']
            genpi, gendelta = pi1pi2_to_pidelta(genpi1, genpi2)
            title += "\n"
            title += fr"Assumed($\pi={genpi:.3f}$ , $\xi={self.ass_pars['xi']:.3f}$ , "
            title += fr"$\delta={gendelta:.3f}$)"
        #TODO: add diss_gen
        # if self.diss_gen is not None:
        #     title += "\n"
        #     title += fr"Assumed($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
        ax.set_title(title)

        R = choices(self.m)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability")

        #p = pmf(m=self.m, pi1=self.pi1, pi2=self.pi2, xi=self.xi, sh=self.sh)
        ax.plot(R, self.theoric, ".b:",
            label="estimated", ms=10)
        if kind == "bar":
            ax.bar(R, self.f/self.n,
                facecolor="None",
                edgecolor="k",
                label="observed")
        else:
            if kind != "scatter":
                print(f"WARNING: kind `{kind}` unknown. Using `scatter` instead.")
            ax.scatter(R, self.f/self.n,
                facecolor="None",
                edgecolor="k", s=200,
                label="observed")
        if self.ass_pars is not None:
            pi1_gen, pi2_gen = self.ass_pars["pi1"], self.ass_pars["pi2"]
            xi_gen = self.ass_pars['xi']
            p_gen = pmf(m=self.m, pi1=pi1_gen, pi2=pi2_gen, xi=xi_gen, sh=self.sh)
            ax.stem(R, p_gen, linefmt="--r",
                markerfmt="none", label="assumed")

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax

    #TODO: add displacement from CUB with no shelter effect
    def _plot_confell(self,
        figsize=(7, 5),
        ci=.95,
        equal=True,
        magnified=False,
        ax=None,
        saveas=None,
        confell=False, debug=False
        ):
        """
        :DEPRECATED:
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        else:
            fig = None

        if equal:
            ax.set_aspect("equal")

        pi = self.estimates[3]
        xi = self.estimates[4]
        delta = self.estimates[5]
        ax.set_xlabel(r"$(1-\pi)$  uncertainty")
        ax.set_ylabel(r"$(1-\xi)$  feeling")

        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)

        ax.plot(1-pi, 1-xi, 
            ".b",ms=20, alpha=.5,
            label="estimated")
        ax.text(1-pi, 1-xi,
            fr"  $\delta = {delta:.3f}$" "\n",
            ha="left", va="bottom")
        if self.ass_pars is not None:
            pass
            #TODO: add ass_pars?
        
        # Confidence Ellipse
        if confell:
            Vpxd = varcov_pxd(
                self.m, self.sh, pi, xi, 
                delta, self.n)
            if debug:
                print()
                print("VARCOV(pxd)")
                print(Vpxd)
                espxd = np.sqrt(
                    np.diag(Vpxd))
                print()
                print("ES(pxd)")
                print(espxd)
            
            conf_ell(
                 Vpxd[:2,:2],
                 1-pi, 1-xi,
                 ci, ax
            )

        if not magnified:
            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            ticks = np.arange(0, 1.05, .1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if xlim[0] < 0:
                ax.set_xlim((0, xlim[1]))
                xlim = ax.get_xlim()
            if xlim[1] > 1:
                ax.set_xlim((xlim[0], 1))
            if ylim[0] < 0:
                ax.set_ylim((0, ylim[1]))
                ylim = ax.get_ylim()
            if ylim[1] > 1:
                ax.set_ylim((ylim[0], 1))
            # ax.axline(
            #     [1-self.pi, 1-self.xi],
            #     slope=self.rho, ls="--"
            # )

        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        ax.grid(visible=True)

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas,
                    bbox_inches='tight')
        return fig, ax

    def plot3d(self, ax, ci=.95,
        magnified=False):
        r"""Plots the estimated parameter values in the parameter space and
        the asymptotic confidence ellipsoid with its projections.
        
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipsoid
        :type ci: float
        :param magnified: if False the limits will be the entire parameter space, otherwise let matplotlib choose the limits
        :type magnified: bool
        :param ax: matplotlib axis, if None a new figure will be created, defaults to None
        :type ax: matplolib ax, optional
        """
        pi = self.estimates[3]
        xi = self.estimates[4]
        de = self.estimates[5]
        V = varcov_pxd(
            self.m, self.sh, pi, xi,
            de, self.n)
        #print()
        #print("VARCOV(pxd)")
        #print(V)
        #espxd = np.sqrt(
        #            np.diag(V))
        #print()
        #print("ES(pxd)")
        #print(espxd)
        plot_ellipsoid(V=V,
            E=(1-pi,1-xi,de), ax=ax,
            zlabel=r"Shelter Choice $\delta$",
            magnified=magnified, ci=ci
        )
        

    def plot(self,
        ci=.95,
        saveas=None,
        confell=False,
        debug=False,
        test3=True,
        figsize=(7, 15)
        ):
        r"""Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipsoid
        :type ci: float
        :param confell: **DEPRECATED**, defaults to False
        :type confell: bool
        :param test3: **DEPRECATED**, defaults to True
        :type test3: bool
        :param debug: **DEPRECATED**, defaults to False
        :type debug: bool
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        fig, ax = plt.subplots(3, 1,
            figsize=figsize,
            constrained_layout=True)
        self.plot_ordinal(ax=ax[0])
        if test3:
            ax[1].remove()
            ax[2].remove()
            ax[1] = fig.add_subplot(3,1,2,
                projection='3d')
            ax[2] = fig.add_subplot(3,1,3,
                projection='3d')
            self.plot3d(ax=ax[1], ci=ci)
            self.plot3d(ax=ax[2], ci=ci,
                magnified=True)
        else:
            self._plot_confell(ci=ci, ax=ax[1],
                confell=confell, debug=debug)
            pi1 = self.estimates[0]
            pi2 = self.estimates[1]
            # self.plot_confell(
            #     ci=ci, ax=ax[2],
            #     magnified=True, equal=False)
            plot_simplex([(pi1, pi2)], ax=ax[2])
            plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
