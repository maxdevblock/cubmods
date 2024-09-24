# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cubeywz-module:

CUB models in Python.
Module for CUBE (Combination of Uniform
and Beta-Binomial) with covariates.

Description:
============
    This module contains methods and classes
    for CUB_YWZ model family.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    :math:`\Pr(R_i=r|\pmb{\theta}) = \pi \beta e(\xi,\phi)+\dfrac{1-\pi}{m},\; r=1,2,\ldots,m`

    :math:`\pi_i = \dfrac{1}{1+e^{-\pmb y_i \pmb \beta}}`

    :math:`\xi_i = \dfrac{1}{1+e^{-\pmb w_i \pmb \gamma}}`

    :math:`\phi_i = e^{\pmb z_i \pmb \alpha}`

Manual and Examples
==========================
    - `Models manual <manual.html#cube-with-covariates>`__


References:
===========
    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        iannario2014modelling
        iannario2015detecting
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

List of TODOs:
===================
  - Manual and Examples

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
from .cube import (
    init_theta as ini_cube, betar
)
from .cub_0w import init_gamma
from .general import (
    logis, choices, colsof, hadprod,
    addones,
    #lsat,
    luni, aic, bic,
    freq, dissimilarity,
    #lsatcov
)
from .smry import CUBres, CUBsample

def pmfi(m, beta, gamma, alpha, Y, W, Z):
    r"""Probability distribution for each subject of a specified CUBE model 
    with covariates.
    
    Auxiliary function of ``.draw()``.

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param alpha: array :math:`\pmb \alpha` of parameters for the overdispersion, whose length equals 
        ``Z.columns.size+1`` to include an intercept term in the model (first entry)
    :type alpha: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    n = W.shape[0]
    pi = logis(Y, beta)
    xi = logis(W, gamma)
    phi = logis(Z, alpha)
    p = np.ndarray(shape=(n, m))
    for i in range(n):
        pBe = betar(m=m, xi=xi[i],
            phi=phi[i])
        p[i,:] = pi[i]*(pBe-1/m) + 1/m
    return p

def pmf(m, beta, gamma, alpha, Y, W, Z):
    r"""Average probability distribution of a specified CUB model 
    with covariates for the feeling component.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param alpha: array :math:`\pmb \alpha` of parameters for the overdispersion, whose length equals 
        ``Z.columns.size+1`` to include an intercept term in the model (first entry)
    :type alpha: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
    :return: the array of the average probability distribution
    :rtype: numpy array
    """
    p = pmfi(m, beta, gamma, alpha,
        Y, W, Z).mean(axis=0)
    return p

def draw(m, beta, gamma, alpha,
    df, formula,
    Y, W, Z, seed=None):
    r"""Draw a random sample from a specified CUBE model.

    :param m: number of ordinal categories
    :type m: int
    :param n: number of ordinal responses to be drawn
    :type n: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param alpha: array :math:`\pmb \alpha` of parameters for the overdispersion, whose length equals 
        ``Z.columns.size+1`` to include an intercept term in the model (first entry)
    :type alpha: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
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
    assert len(alpha) == Z.shape[1]+1
    assert Y.shape[0] == W.shape[0] == Z.shape[0]
    n = W.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, beta=beta,
        gamma=gamma, alpha=alpha,
        Y=Y, W=W, Z=Z)
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
    theoric = pmf(m=m, beta=beta,
        gamma=gamma, alpha=alpha,
        Y=Y, W=W, Z=Z)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        beta, gamma, alpha
    ))
    par_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        Z.columns,
    ))
    sample = CUBsample(
        model="CUBE(YWZ)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, diss=diss,
        theoric=theoric, df=df,
        formula=formula
    )
    return sample

def init_theta(m, sample, W, p, v):
    r"""Preliminary parameter estimates for CUBE models with covariates.

    Compute preliminary parameter estimates for a CUBE model with covariates for all the three parameters. 
    These estimates are set as initial values to start the E-M algorithm within maximum likelihood estimation.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param p: number of covariates for the uncertainty component
    :type p: int
    :param v: number of covariates for the overdispersion
    :type v: int
    :return: a tuple of :math:`(\pmb \beta^{(0)}, \pmb \gamma^{(0)}, \pmb \alpha^{(0)})` of preliminary estimates of parameter vectors for 
        :math:`\pi = \pi(\pmb{\beta})`, \; \xi=\xi(\pmb{\gamma}),\; \phi=\phi(\pmb{\alpha})` respectively, of a CUBE model with covariates for all the three
        parameters. In details, they have length equal to ``Y.columns.size+1``, ``W.columns.size+1`` and
        ``Z.columns.size+1``, respectively, to account for an intercept term for each component.
    :rtype: tuple of arrays
    """
    gamma = init_gamma(sample=sample, m=m, W=W)
    pi, _, _ = ini_cube(sample=sample, m=m)
    beta0 = np.log(pi/(1-pi))
    beta = np.concatenate((
        [beta0], np.repeat(.1, p)
    ))
    alpha0 = np.log(.1)
    alpha = np.concatenate((
        [alpha0],
        np.repeat(.1, v)
    ))
    return beta, gamma, alpha

def betabinomial(m, sample, xi, phi):
    r"""Beta-Binomial probabilities of ordinal responses, with feeling and overdispersion parameters
    for each observation.

    Compute the Beta-Binomial probabilities of ordinal responses, given feeling and overdispersion
    parameters for each observation.

    The Beta-Binomial distribution is the Binomial distribution in which the probability of success at
    each trial is random and follows the Beta distribution. It is frequently used in Bayesian 
    statistics, empirical Bayes methods and classical statistics as an overdispersed binomial distribution. 

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :return: array of the same length as ``sample``, containing the Beta-Binomial probabilities of each observation,
        for the corresponding feeling and overdispersion parameters.
    :rtype: array
    """
    n = sample.size
    p = np.repeat(np.nan, n)
    sample = np.array(sample).astype(int)
    for i in range(n):
        b = betar(m=m, xi=xi[i], phi=phi[i])
        p[i] = b[sample[i]-1]
    return p

def loglik(m, sample, Y, W, Z,
    beta, gamma, alpha
    ):
    r"""Log-likelihood function of a CUBE model with covariates.

    Compute the log-likelihood function of a CUBE model for ordinal responses,
    with covariates for explaining all the three parameters.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param alpha: array :math:`\pmb \alpha` of parameters for the overdispersion, whose length equals 
        ``Z.columns.size+1`` to include an intercept term in the model (first entry)
    :type alpha: array of float
    :return: the log-likelihood value
    :rtype: float
    """
    pi = logis(Y, beta)
    xi = logis(W, gamma)
    phi = 1/(-1+1/logis(Z, alpha))
    b = betabinomial(m=m, sample=sample,
        xi=xi, phi=phi)
    p = pi*(b-1/m)+1/m
    l = np.sum(np.log(p))
    return l

def Quno(beta, esterno1):
    r"""Auxiliary function for the log-likelihood estimation of CUBE models with covariates.

    Define the opposite one of the two scalar functions that are maximized when running the E-M algorithm
    for CUBE models with covariates for feeling, uncertainty and overdispersion.

    It is iteratively called as an argument of "optim" within CUBE function (with covariates) as  the function
    to minimize to compute the maximum likelihood estimates for the feeling and the overdispersion components. 

    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param esterno1: matrix binding together the column vector of the posterior probabilities
        that each observed rating has been generated by the first component distribution of the mixture, with the matrix 
        :math:`\pmb y` of explicative  variables for the uncertainty component, expanded with a unitary vector in the first column to 
        consider also an intercept term
    :type esterno1: ndarray
    """
    tauno = esterno1[:,0]
    covar = esterno1[:,1:]
    ybeta = covar @ beta
    r = -np.sum(
        tauno*ybeta
        -np.log(1+np.exp(ybeta))
    )
    return r

def Qdue(pars, tauno, sample, W, Z, m):
    r"""Auxiliary function for the log-likelihood estimation of CUBE models with covariates.

    Define the opposite of one of the two scalar functions that are maximized when running the E-M 
    algorithm for CUBE models with covariates for feeling, uncertainty and overdispersion.

    :param pars: array of initial estimates of parameters for the feeling component and the overdispersion effect
    :type pars: array
    :param tauno: the column vector of the posterior probabilities that each observed rating
        has been generated by the distribution of the first component of the mixture
    :type tauno: array
    :param sample: array of ordinal responses
    :type sample: array of int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
    :param m: number of ordinal categories
    :type m: int
    """
    #v = esterno2.shape[1]-q-2
    #print(f"e2:{esterno2.shape}")
    # tauno = esterno2[:,0]
    # sample = esterno2[:,1]
    # W = esterno2[:,2:-v]#o +2?
    # Z = esterno2[:,-v:]
    #print(f"pars:{pars.shape}")
    pz = colsof(Z)+1
    gamma = pars[:-pz]
    alpha = pars[-pz:]
    #print(f"W:{W.shape}, gamma:{gamma.shape}")
    xi = logis(W, gamma)
    phi = 1/(-1+1/logis(Z, alpha))
    betabin = betabinomial(m=m,
        sample=sample,
        xi=xi, phi=phi)
    r = -np.sum(tauno*np.log(betabin))
    return r

def auxmat(m, xi, phi, a,b,c,d,e):
    r"""Auxiliary matrix.

    Returns an auxiliary matrix needed for computing the variance-covariance matrix of a CUBE model with covariates.

    :param m: number of ordinal categories
    :type m: int
    :param xi: feeling parameters :math:`\pmb\xi`
    :type xi: array of float
    :param phi: overdispersion parameter :math:`\pmb\phi`
    :type phi: array of float
    :param a,b,c,d,e: see the reference paper *DOI: 10.1080/03610926.2013.821487* for details
    :type a,b,c,d,e: float
    """
    elemat = np.ndarray(shape=(m, xi.size))
    elemat[:] = np.nan
    R = choices(m)
    for r in R:
        o = r-1
        elemat[o,:] = (
            e*(o**d) /
            ((a+b*xi+phi*o)**c)
        )
    return elemat

def varcov(m, sample, beta, gamma, alpha,
    Y, W, Z):
    r"""Variance-covariance matrix of a CUBE model with covariates.

    Compute the variance-covariance matrix of parameter estimates of a CUBE model with covariates
    for all the three parameters.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param alpha: array :math:`\pmb \alpha` of parameters for the overdispersion, whose length equals 
        ``Z.columns.size+1`` to include an intercept term in the model (first entry)
    :type alpha: array of float
    :return: the variance-covariance matrix
    :rtype: ndarray
    """
    n = sample.size
    # p = colsof(Y)
    # q = colsof(W)
    # v = colsof(Z)
    #npars = beta.size+gamma.size+alpha.size
    pi = logis(Y, beta)
    xi = logis(W, gamma)
    phi = 1/(1/logis(Z, alpha)-1)
    pBe = betabinomial(m=m, sample=sample,
        xi=xi, phi=phi)
    probi = pi*(pBe-1/m)+1/m
    uui = 1-1/(m*probi)
    ubari = uui+pi*(1-uui)

    mats1 = auxmat(m,xi,phi,1,-1,1,0, 1)
    mats2 = auxmat(m,xi,phi,0, 1,1,0, 1)
    mats3 = auxmat(m,xi,phi,1,-1,1,1, 1)
    mats4 = auxmat(m,xi,phi,0, 1,1,1, 1)
    mats5 = auxmat(m,xi,phi,1, 0,1,1, 1)

    matd1 = auxmat(m,xi,phi,1,-1,2,0, 1)
    matd2 = auxmat(m,xi,phi,0, 1,2,0,-1)
    matd3 = auxmat(m,xi,phi,1,-1,2,1, 1)
    matd4 = auxmat(m,xi,phi,0, 1,2,1,-1)

    math3 = auxmat(m,xi,phi,1,-1,2,2,-1)
    math4 = auxmat(m,xi,phi,0, 1,2,2,-1)
    math5 = auxmat(m,xi,phi,1, 0,2,2,-1)

    #print("uui"); print(uui)
    #print("ubari"); print(ubari)
    #with np.printoptions(
    #    precision=5, suppress=True):
    #    print("mats5"); print(mats5)

    #TODO: in R è m ma dev'essere n! Perchè in R si aggiunge, in Python no.
    S1 = np.repeat(np.nan, n)
    S2 = np.repeat(np.nan, n)
    S3 = np.repeat(np.nan, n)
    S4 = np.repeat(np.nan, n)
    D1 = np.repeat(np.nan, n)
    D2 = np.repeat(np.nan, n)
    D3 = np.repeat(np.nan, n)
    D4 = np.repeat(np.nan, n)
    H3 = np.repeat(np.nan, n)
    H4 = np.repeat(np.nan, n)
    
    sam = sample - 1
    for i in range(n):
        #occhio agli indici...!
        #print(f"i:{i}, sam[i]:{sam[i]}")
        #print(f"mats1:{mats1.shape}")
        #print(f"S1:{S1.shape}")
        #print(f"S1[i]:{S1[i]}")
        S1[i] = mats1[:sample[i],i].sum()-\
            mats1[sam[i],i]
        S2[i] = mats2[:m-sample[i]+1,i].sum()-\
            mats2[m-sample[i],i]
        #print(mats2[:m-sample[i]+1,i])
        #print(mats2[m-sample[i],i])
        S3[i] = mats3[:sample[i],i].sum()-\
            mats3[sam[i],i]
        S4[i] = mats4[:m-sample[i]+1,i].sum()-\
            mats4[m-sample[i],i]
        D1[i] = matd1[:sample[i],i].sum()-\
            matd1[sam[i],i]
        D2[i] = matd2[:m-sample[i]+1,i].sum()-\
            matd2[m-sample[i],i]
        D3[i] = matd3[:sample[i],i].sum()-\
            matd3[sam[i],i]
        D4[i] = matd4[:m-sample[i]+1,i].sum()-\
            matd4[m-sample[i],i]
        H3[i] = math3[:sample[i],i].sum()-\
            math3[sam[i],i]
        H4[i] = math4[:m-sample[i]+1,i].sum()-\
            math4[m-sample[i],i]
    #fino (m-1) ???
    S5 = mats5[:m-1].sum(axis=0)
    H5 = math5[:m-1].sum(axis=0)
    #print("S1"); print(S1)
    #print("S5"); print(S5)

    CC = S2-S1
    DD = D2-D1
    EE = S3+S4-S5
    FF = D3+D4
    GG = H3+H4-H5

    vibe = uui*(1-pi)
    viga = ubari*xi*(1-xi)*CC
    vial = ubari*phi*EE
    ubebe = uui*(1-pi)*(1-2*pi)
    ugabe = ubari*xi*(1-xi)*(1-pi)*CC
    ualbe = ubari*phi*(1-pi)*EE
    ugaga = ubari*xi*(1-xi)*((1-2*xi)*CC\
            + xi*(1-xi)*(CC**2+DD))
    ualga = ubari*phi*xi*(1-xi)*(
            FF+CC*EE)
    ualal = ubari*phi*(EE+phi*(EE**2+GG))

    #print(vibe)
    gbebe = vibe*vibe-ubebe
    ggabe = viga*vibe-ugabe
    galbe = vial*vibe-ualbe
    ggaga = viga*viga-ugaga
    galga = vial*viga-ualga
    galal = vial*vial-ualal

    #print("gbebe"); print(gbebe)

    YY = addones(Y)
    WW = addones(W)
    ZZ = addones(Z)

    infbebe = YY.T @ hadprod(YY,gbebe)
    infgabe = WW.T @ hadprod(YY,ggabe)
    infalbe = ZZ.T @ hadprod(YY,galbe)
    infgaga = WW.T @ hadprod(WW,ggaga)
    infalga = ZZ.T @ hadprod(WW,galga)
    infalal = ZZ.T @ hadprod(ZZ,galal)
    infbega = infgabe.T
    infbeal = infalbe.T
    infgaal = infalga.T

    #print("mats2"); print(mats2)
    #print("S2"); print(S2)
    #print("viga"); print(viga)
    #print("vibe"); print(vibe)
    #print("ugabe"); print(ugabe)
    #print("ggabe"); print(ggabe)
    #print("infalbe"); print(infalbe)
    #print("infalga"); print(infalga)
    #print("infalal"); print(infalal)

    npi =  colsof(YY)
    nxi =  colsof(WW)
    nphi = colsof(ZZ)
    npars = npi+nxi+nphi
    matinf = np.ndarray(shape=(npars,npars))
    matinf[:] = np.nan
    
    for i in range(npi):
        #print("i"); print(i)
        matinf[i,:] = np.r_[
            infbebe[i,:],
            infbega[i,:],
            infbeal[i,:]
        ]
    for i in np.arange(npi,npi+nxi):
        #print("i"); print(i)
        #print("i-npi"); print(i-npi)
        matinf[i,:] = np.r_[
            infgabe[i-npi,:],
            infgaga[i-npi,:],
            infgaal[i-npi,:]
        ]
    for i in np.arange(npi+nxi, npars):
        #print(i, npi, nxi)
        #print("i"); print(i)
        #print("i-npi-nxi"); print(i-npi-nxi)
        matinf[i,:] = np.r_[
            infalbe[i-npi-nxi,:],
            infalga[i-npi-nxi,:],
            infalal[i-npi-nxi,:]
        ]
    #print("inf"); print(matinf)
    varmat = np.ndarray(shape=(npars,npars))
    varmat[:] = np.nan
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(matinf)
    return varmat
       
def mle(m, sample, Y, W, Z, df, formula,
    ass_pars=None,
    maxiter=1000, tol=1e-2):
    r"""Main function for CUBE models with covariates.

    Function to estimate and validate a CUBE model with 
    explicative covariates for all the three parameters.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param Z: dataframe of covariates for explaining the overdispersion
    :type Z: pandas dataframe
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
    :return: an instance of ``CUBresCUBEYWZ`` (see the Class for details)
    :rtype: object
    """
    start = dt.datetime.now()
    f = freq(m=m, sample=sample)
    n = sample.size
    YY = addones(Y)
    # WW = addones(W)
    # ZZ = addones(Z)
    p = colsof(Y)
    q = colsof(W)
    v = colsof(Z)
    
    beta, gamma, alpha = init_theta(
        m=m, sample=sample, W=W, p=p, v=v)
    #print(f"beta:{beta}")
    #print(f"alpha:{alpha}")
    #print(f"gamma:{gamma}")
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    
    niter = 1
    while niter < maxiter:
        pi = logis(Y=Y, param=beta)
        xi = logis(Y=W, param=gamma)
        phi = 1/(
            1/logis(Y=Z, param=alpha)-1)
        betabin = betabinomial(m=m,
            #rank o sample?
            # sample=rank,
            sample=sample,
            xi=xi, phi=phi)
        probi = pi*(betabin-1/m)+1/m
        lold = np.sum(np.log(probi))
        taui = 1-(1-pi)/(m*probi)
        esterno1 = np.c_[taui, YY]
        #esterno2 = np.c_[taui, sample, W, Z]
        pars = np.concatenate((
            gamma, alpha
        ))
        #print(f"it:{niter}*************")
        #print("Quno ->")
        optimbeta = minimize(
            Quno, x0=beta,
            args=esterno1,
            #method="Nelder-Mead"
        )
        #print("Quno -|\nQdue ->")
        optimpars = minimize(
            Qdue, x0=pars,
            args=(taui, sample, W, Z, m),
            #method="Nelder-Mead"
        )
        #print("Qdue -|")
        beta = optimbeta.x
        gamma = optimpars.x[:(q+1)]#o p?
        alpha = optimpars.x[(q+1):]
        l = loglik(m=m, sample=sample,
            Y=Y, W=W, Z=Z, 
            beta=beta, gamma=gamma,
            alpha=alpha)
        testl = abs(l-lold)
        #print(f"test:{testl}")
        if testl <= tol:
            break
        #else:
        #    lold = l
        niter += 1
    varmat = varcov(m, sample, beta, gamma, alpha,
    Y, W, Z)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.concatenate((
        beta, gamma, alpha
    ))
    est_names = np.concatenate((
        ["constant"],
        [x for x in Y.columns],
        ["constant"],
        [x for x in W.columns],
        ["constant"],
        [x for x in Z.columns],
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        [None for _ in range(p)],
        ["Feeling"],
        [None for _ in range(q)],
        ["Overdispersion"],
        [None for _ in range(v)]
    ))
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    muloglik = l/n
    #logliksat = lsat(f=f, n=n)
    #logliksatcov = lsatcov(
    #        sample=sample,
    #        covars=[Y,W,Z]
    #    )
    loglikuni = luni(m=m, n=n)
    AIC = aic(l=l, p=estimates.size)
    BIC = bic(l=l, p=estimates.size, n=n)
    theoric = pmf(m, beta, gamma,
        alpha, Y, W, Z)
    diss = dissimilarity(f/n, theoric)
    #dev = 2*(logliksat-l)
    end = dt.datetime.now()
    
    return CUBresCUBEYWZ(
        model="CUBE(YWZ)",
        m=m, sample=sample, n=n,
        niter=niter, maxiter=maxiter,
        tol=tol, f=f,
        estimates=estimates,
        est_names=est_names,
        e_types=e_types,
        stderrs=stderrs,
        varmat=varmat,
        wald=wald, pval=pval,
        AIC=AIC, BIC=BIC,
        loglike=l, muloglik=muloglik,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        #dev=dev,
        theoric=theoric, diss=diss,
        seconds=(end-start).total_seconds(),
        time_exe=start,
        ass_pars=ass_pars,
        df=df, formula=formula
    )

class CUBresCUBEYWZ(CUBres):
    """Object returned by ``.mle()`` function.
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
        #     title += fr"Gener($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
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
        #     markerfmt="none", label="generating")

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
