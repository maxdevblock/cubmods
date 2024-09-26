# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace, invalid-unary-operand-type
r"""
.. _cubshywx-module:

CUB models in Python.
Module for CUBSH (Combination of Uniform
and Binomial with Shelter Effect) with covariates.

Description:
============
    This module contains methods and classes
    for CUBSH_YWX model family.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    :math:`\Pr(R_i=r|\pmb\theta;\pmb y_i; \pmb w_i; \pmb x_i) = \delta_i D_r^{(c)} + (1 - \delta_i)[ \pi_i b_r(\xi_i) + (1-\pi_i)/m ],\; r=1,2,\ldots,m`

    :math:`\xi_i = \dfrac{1}{1+e^{-\pmb w_i \pmb\gamma}}`

    :math:`\pi_i = \dfrac{1}{1+e^{-\pmb y_i \pmb\beta}}`

    :math:`\delta_i = \dfrac{1}{1+e^{-\pmb x_i \pmb\omega}}`


Manual, Examples and References:
===============================
    - `Models manual <manual.html#cubsh-with-covariates>`__
  
List of TODOs:
==============
  - ...

Credits
==============
    :Author:      Massimo Pierini

    :Date:        2023-24
    :Credits:     Domenico Piccolo, Rosaria Simone
    :Contacts:    cub@maxpierini.it

Classes and Functions
=====================
"""

import datetime as dt
import numpy as np
#import pandas as pd
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from .cub import (
    init_theta as inipixi
)
from .cub_0w import (
    init_gamma, bitgamma
)
from .general import (
    choices, freq, logis, colsof,
    addones, hadprod, aic, bic,
    #lsat, 
    luni, dissimilarity,
    #lsatcov
)
from .cubsh import (
    pmf as pmf_cubsh,
    pidelta_to_pi1pi2
)
from .smry import CUBres, CUBsample

def pmf(m, sh, beta, gamma, omega,
    Y, W, X):
    r"""Average probability distribution of a specified CUBSH model with covariates
    (aka GeCUB model).

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :return: the probability distribution
    :rtype: array
    """
    p = pmfi(m, sh, beta, gamma, omega,
        Y, W, X)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, sh, beta, gamma, omega,
    Y, W, X):
    r"""Probability distribution for each subject of a specified CUBSH model with covariates
    (aka GeCUB model).

    Auxiliary function of ``.draw()``.

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    pi = logis(Y, beta)
    xi = logis(W, gamma)
    delta = logis(X, omega)
    pi1, pi2 = pidelta_to_pi1pi2(pi, delta)
    n = Y.shape[0]
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = pmf_cubsh(
            m=m, sh=sh,
            pi1=pi1[i], pi2=pi2[i],
            xi=xi[i]
        )
    return p

def draw(m, sh, beta, gamma, omega,
    Y, W, X, df, formula, seed=None):
    r"""Draw a random sample from a specified CUBSH model with covariates
    (aka GeCUB model).

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :param n: number of ordinal responses to be drawn
    :type n: int
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    #np.random.seed(seed)
    assert len(beta) == Y.shape[1]+1
    assert len(gamma) == W.shape[1]+1
    assert len(omega) == X.shape[1]+1
    assert Y.shape[0] == W.shape[0] == X.shape[0]
    n = W.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, sh=sh, beta=beta,
        gamma=gamma, omega=omega,
        Y=Y, W=W, X=X)
    #print("n", n)
    for i in range(n):
        if seed is not None:
            np.random.seed(seed*i)
        rv[i] = np.random.choice(
            choices(m=m),
            size=1,
            replace=True,
            p=theoric_i[i]
        )
    f = freq(m=m, sample=rv)
    theoric = theoric_i.mean(axis=0)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        beta, gamma, omega
    ))
    par_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        X.columns,
    ))
    p_types = np.concatenate((
        np.repeat(["Uncertainty"], len(beta)),
        np.repeat(["Feeling"], len(gamma)),
        np.repeat(["Shelter"], len(omega)),
    ))
    sample = CUBsample(
        model="CUBSH(YWX)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        p_types=p_types,
        seed=seed, diss=diss,
        theoric=theoric, df=df,
        formula=formula, sh=sh
    )
    return sample

def init_theta(m, sample, p, s, W):
    r"""Preliminary estimators for CUBSH models with covariates.

    Computes preliminary parameter estimates of a CUBSH model without covariates for given ordinal
    responses. These preliminary estimators are used within the package code to start the E-M algorithm.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param p: number of covariates for the uncertainty component
    :type p: int
    :param s: number of covariates for the shelter effect
    :type s: int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: a tuple of :math:`(\pmb \beta^{(0)}, \pmb \gamma^{(0)}, \pmb \omega^{(0)})` of preliminary estimates of parameter vectors for 
        :math:`\pi = \pi(\pmb{\beta})`, \; \xi=\xi(\pmb{\gamma}),\; \delta=\delta(\pmb{\omega})` respectively, of a CUBSH model with covariates for all the three
        parameters. In details, they have length equal to ``Y.columns.size+1``, ``W.columns.size+1`` and
        ``X.columns.size+1``, respectively, to account for an intercept term for each component.
    :rtype: tuple of arrays
    """
    f = freq(m=m, sample=sample)
    pi, _ = inipixi(f=f, m=m)
    beta0 = np.log(pi/(1-pi))
    beta = np.concatenate((
        [beta0], np.repeat(0., p)
    ))
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    gamma = init_gamma(sample=sample,
        m=m, W=W)
    omega = np.repeat(.1, s+1)
    return beta, gamma, omega

def prob(m, sample, sh, Y, W, X,
    beta, gamma, omega):
    r"""Probability distribution of a CUBSH model with covariates.

    Compute the probability distribution of a CUBSH model with covariates.

    :math:`\Pr(R_i=r_i|\pmb\theta;\pmb T_i),\; i=1 \ldots n`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :param sample: array of ordinal responses
    :type sample: array of int
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    alpha1 = logis(X, omega)
    alpha2 = (1-alpha1)*logis(Y, beta)
    D = (sample==sh).astype(int)
    bg = bitgamma(sample=sample, m=m, 
        W=W, gamma=gamma)
    d = 1 - alpha1 - alpha2
    p = alpha1*D + alpha2*bg + d/m
    return p

def varcov(sample, m, sh, Y, W, X,
    beta, gamma, omega):
    r"""Variance-covariance matrix of a CUBSH model with covariates

    Compute the variance-covariance matrix of parameter estimates of a CUBSH model with covariates.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :param sample: array of ordinal responses
    :type sample: array of int
    :return: the variance-covariance matrix of the model
    :rtype: numpy ndarray
    """
    probi = prob(m, sample, sh, Y, W, X,
        beta, gamma, omega)
    vvi = 1/probi
    D = (sample==sh).astype(int)
    pii = logis(Y, beta)
    xii = logis(W, gamma)
    deltai = logis(X, omega)
    bri = bitgamma(sample=sample, m=m,
        W=W, gamma=gamma)
    YY = addones(Y)
    WW = addones(W)
    XX = addones(X)
    npar = colsof(YY)+colsof(WW)+colsof(XX)
    mi = m-sample-(m-1)*xii
    vAA = pii*(1-pii)*(1-deltai)*(bri-1/m)*vvi
    AA = hadprod(YY, vAA)
    vBB = pii*(1-deltai)*mi*bri*vvi
    BB = hadprod(WW, vBB)
    vCC = deltai*(D-probi)*vvi
    CC = hadprod(XX, vCC)
    
    di = pii*(1-pii)*(1-2*pii)*(1-deltai)*(bri-1/m)*vvi
    gi = pii*(1-deltai)*bri*(mi**2-(m-1)*xii)*(1-xii)*vvi
    li = deltai*(1-2*deltai)*(D-probi)*vvi
    ei = pii*(1-pii)*(1-deltai)*bri*mi*vvi
    fi = (-pii)*(1-pii)*deltai*(1-deltai)*(bri-1/m)*vvi
    hi = (-pii)*deltai*(1-deltai)*bri*mi*vvi
    
    i11 = (AA.T @ AA) - (YY.T @ hadprod(YY,di))
    i22 = (BB.T @ BB) - (WW.T @ hadprod(WW,gi))
    i33 = (CC.T @ CC) - (XX.T @ hadprod(XX,li))
    i21 = (BB.T @ AA) - (WW.T @ hadprod(YY,ei))
    i31 = (CC.T @ AA) - (XX.T @ hadprod(YY,fi))
    i32 = (CC.T @ BB) - (XX.T @ hadprod(WW,hi))
    i12 = i21.T
    i13 = i31.T
    i23 = i32.T
    matinf = np.r_[
        np.c_[i11, i12, i13],
        np.c_[i21, i22, i23],
        np.c_[i31, i32, i33]
    ]
    #print(matinf)
    varmat = np.ndarray(shape=(npar, npar))
    varmat[:] = np.nan
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        varmat = np.linalg.inv(matinf)
    return varmat

def loglik(m, sample, sh, Y, W, X,
    beta, gamma, omega):
    r"""Log-likelihood function of a CUBSH model with covariates.

    Compute the log-likelihood function of a CUBE model for ordinal responses,
    with covariates for explaining all the three parameters (GeCUB model).

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :param sample: array of ordinal responses
    :type sample: array of int
    :return: the log-likelihood value
    :rtype: float
    """
    p = prob(m, sample, sh, Y, W, X,
        beta, gamma, omega)
    l = np.sum(np.log(p))
    return l

def Q1(param, dati1, p):
    r"""Auxiliary function for the log-likelihood estimation of GeCUB models.

    Define the opposite one of the two scalar functions that are maximized when running the E-M algorithm
    for GeCUB models with covariates for feeling, uncertainty and shelter effect.

    :param param: array of initial estimates of parameters for the uncertainty component
    :type param: array
    :param dati1: auxiliary matrix
    :type dati1: ndarray or dataframe
    :param p: number of covariates for the uncertainty component
    :type p: int
    """
    omega = param[:-(p+1)]
    beta = param[-(p+1):]
    tau1 = dati1[0]
    tau2 = dati1[1]
    tau3 = 1 - tau1 - tau2
    X = dati1[2]
    Y = dati1[3]
    alpha1 = logis(X, omega)
    alpha2 = (1-alpha1)*logis(Y, beta)
    alpha3 = 1 - alpha1 - alpha2
    esse1 = (tau1*np.log(alpha1)).sum()
    esse2 = (tau2*np.log(alpha2)).sum()
    esse3 = (tau3*np.log(alpha3)).sum()
    esse = -(esse1+esse2+esse3)
    return esse

def Q2(param, dati2, m):
    r"""Auxiliary function for the log-likelihood estimation of GeCUB models.

    Define the opposite one of the two scalar functions that are maximized when running the E-M algorithm
    for GeCUB models with covariates for feeling, uncertainty and shelter effect.

    :param param: array of initial estimates of parameters for the feeling component
    :type param: array
    :param dati2: auxiliary matrix
    :type dati2: ndarray or dataframe
    :param m: number of ordinal categories
    :type m: int
    """
    tau2 = dati2[0]
    sample = dati2[1]
    W = dati2[2]
    bg = bitgamma(sample=sample, m=m, 
        W=W, gamma=param)
    return -(tau2*np.log(bg)).sum()

def mle(m, sample, sh, Y, W, X,
    df, formula, ass_pars=None,
    maxiter=500, tol=1e-4):
    r"""Main function for CUBSH models with covariates for all the components

    Function to estimate and validate a CUBSH model for given ordinal responses, with covariates for
    explaining all the components and the shelter effect.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
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
    :return: an instance of ``CUBresCUBSHYWZ`` (see the Class for details)
    :rtype: object
    """
    start = dt.datetime.now()
    n = sample.size
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    p = colsof(Y)
    q = colsof(W)
    s = colsof(X)
    beta, gamma, omega = init_theta(
        m, sample, p, s, W)
    psi = np.concatenate((omega, beta))
    #print(omega.size, p, s, psi.shape)
    l = loglik(m, sample, sh, Y, W, X,
        beta, gamma, omega)
    
    niter = 1
    while niter < maxiter:
        lold = l
        alpha1 = logis(X, omega)
        alpha2 = (1-alpha1)*logis(Y, beta)
        alpha3 = 1 - alpha1 - alpha2
        
        p1 = (sample==sh).astype(int)
        p2 = bitgamma(sample=sample, m=m, 
            W=W, gamma=gamma)
        p3 = 1/m
        
        num1 = alpha1*p1
        num2 = alpha2*p2
        num3 = alpha3*p3
        den = num1+num2+num3
        
        ttau1 = num1/den
        ttau2 = num2/den
        #ttau3 = 1 - ttau1 - ttau2
        
        dati1 = [ttau1,ttau2,X,Y]
        dati2 = [ttau2,sample,W]
        
        optim1 = minimize(
            Q1, x0=psi,
            args=(dati1, p)
        )
        optim2 = minimize(
            Q2, x0=gamma,
            args=(dati2, m)
        )
        
        psi = optim1.x
        omega = psi[:-beta.size]
        beta = psi[-beta.size:]
        gamma = optim2.x
        
        l = loglik(m, sample, sh, Y, W, X,
            beta, gamma, omega)
        testl = abs(l-lold)
        if testl <= tol:
            break
        niter += 1
    muloglik = l/n
    varmat = varcov(sample=sample, m=m,
        sh=sh, Y=Y, W=W, X=X, beta=beta,
        gamma=gamma, omega=omega)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.concatenate((
        beta, gamma, omega
    ))
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    AIC = aic(l=l, p=wald.size)
    BIC = bic(l=l, p=wald.size, n=n)
    loglikuni = luni(m=m, n=n)
    f = freq(sample=sample, m=m)
    #logliksat = lsat(n=n, f=f)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[Y,W,X]
    #)
    #dev = 2*(logliksat-l)
    theoric = pmf(m, sh, beta, gamma, omega,
        Y, W, X)
    diss = dissimilarity(f/n, theoric)
    est_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        X.columns
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        [None for _ in range(p)],
        ["Feeling"],
        [None for _ in range(q)],
        ["Shelter effect"],
        [None for _ in range(s)]
    ))
    end = dt.datetime.now()
    
    return CUBresCUBSHYWX(
        model="CUBSH(YWX)",
        m=m, n=n, sh=sh, sample=sample,
        f=f, theoric=theoric,
        niter=niter, maxiter=maxiter,
        tol=tol, stderrs=stderrs,
        est_names=est_names,
        e_types=e_types,
        estimates=estimates,
        wald=wald, pval=pval,
        loglike=l, muloglik=muloglik,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        diss=diss, varmat=varmat,
        #dev=dev,
        AIC=AIC, BIC=BIC,
        seconds=(end-start).total_seconds(),
        time_exe=start, ass_pars=ass_pars,
        df=df, formula=formula
    )
    
class CUBresCUBSHYWX(CUBres):
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        """Plots relative average frequencies of observed sample, estimated average probability distribution and,
        if provided, average probability distribution of a known model.

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
        
        #pi = self.estimates[0]
        #xi = self.estimates[1]
        #phi = self.estimates[2]
        title = "AVERAGE ESTIMATED PROBABILITY\n"
        title += f"{self.model} model    "
        title += f"$n={self.n}$\n"
        #title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$ , $\phi={phi:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.3f}"
        #TODO: add dissimilarity from generating model
        # if self.diss_gen is not None:
        #     title += "\n"
        #     title += fr"Assumed($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
        ax.set_title(title)

        R = choices(self.m)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability")

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
        # if self.ass_pars is not None:
        #     pi_gen = self.ass_pars["pi"]
        #     gamma_gen = self.ass_pars["gamma"]
        #     phi_gen = self.ass_pars["phi"]
        #     p_gen = pmf(m=self.m, pi=pi_gen,
        #         gamma=gamma_gen, phi=phi_gen,
        #         W=self.W)
        #     ax.stem(R, p_gen, linefmt="--r",
        #     markerfmt="none", label="assumed")

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
    
    def plot(self,
        #ci=.95,
        saveas=None,
        figsize=(7, 5)
        ):
        """Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_ordinal(ax=ax)
        #self.plot_confell(ci=ci, ax=ax[1])
        #self.plot_confell(
        #    ci=ci, ax=ax[2],
        #    magnified=True, equal=False)
        #plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax