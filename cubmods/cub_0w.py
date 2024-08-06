# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial) with covariates for the feeling component.

Description:
============
    This module contains methods and classes
    for CUB_0W model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

    :math:`\Pr(R=r_i|\pmb{\theta}_i) = \pi \dbinom{m-1}{r_i-1}(1-\xi_i)^{r_i-1}\xi_i^{m-r_i}+\dfrac{1-\pi}{m}`

    :math:`\xi_i = \dfrac{1}{1+e^{-\pmb w_i \pmb\gamma}}`

Manual and Examples
==========================
  - Manual https://github.com/maxdevblock/cubmods/blob/main/Manual/02_cub_family.md


References:
===========
  - D'Elia A. (2003). Modelling ranks using the inverse hypergeometric distribution, Statistical Modelling: an International Journal, 3, 65--78
  - D'Elia A. and Piccolo D. (2005). A mixture model for preferences data analysis, Computational Statistics & Data Analysis},  \bold{49, 917--937
  - Capecchi S. and Piccolo D. (2017). Dealing with heterogeneity in ordinal responses, Quality and Quantity, 51(5), 2375--2393
  - Iannario M. (2014). Modelling Uncertainty and Overdispersion in Ordinal Data, Communications in Statistics - Theory and Methods, 43, 771--786
  - Piccolo D. (2015). Inferential issues for CUBE models with covariates, Communications in Statistics. Theory and Methods, 44(23), 771--786.
  - Iannario M. (2015). Detecting latent components in ordinal data with overdispersion by means of a mixture distribution, Quality & Quantity, 49, 977--987
  - Iannario M. and Piccolo D. (2016a). A comprehensive framework for regression models of ordinal data. Metron, 74(2), 233--252.
  - Iannario M. and Piccolo D. (2016b). A generalized framework for modelling ordinal data. Statistical Methods and Applications, 25, 163--189.

  
List of TODOs:
==============
  - ...

:Author:      Massimo Pierini
:Institution: Universitas Mercatorum
:Affiliation: Graduand in Statistics & Big Data (L41)
:Date:        2023-24
:Credit:      Domenico Piccolo, Rosaria Simone
:Contacts:    cub@maxpierini.it
"""

import datetime as dt
import numpy as np
#import pandas as pd
#from scipy.special import binom
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, dissimilarity,
    #chisquared, conf_ell,
    bitgamma,
    logis, hadprod, luni, #lsat,
    #lsatcov,
    addones, colsof, aic, bic
)
from . import cub
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def pmf(m, pi, gamma, W):
    r"""Average Probability Mass of a specified CUB model 
    with covariates for the feeling component.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R = r | \pmb\theta_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :return: the vector of the probability distribution.
    :rtype: numpy array
    """
    n = W.shape[0]
    p = pmfi(m, pi, gamma, W)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, pi, gamma, W):
    r"""Probability Mass for each subject of a specified CUB model 
    with covariates for the feeling component.
    
    Auxiliary function of ``.draw()``.

    :math:`\Pr(R = r | \pmb\theta_i),\; i=1 \ldots n \; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :return: the matrix of the probability distribution.
    :rtype: numpy ndarray
    """
    n = W.shape[0]
    xi_i = logis(W, gamma)
    p = np.ndarray(shape=(n, m))
    for i in range(n):
        xi = xi_i[i]
        p[i,:] = cub.pmf(m=m, pi=pi, xi=xi)
    #pr = p.mean(axis=0)
    return p

def prob(sample, m, pi, gamma, W):
    r"""Probability distribution of a CUB model with covariates for the feeling component
    given an observed sample

    Compute the probability distribution of a CUB model with covariates
    for the feeling component, given an observed sample.
    
    :math:`\Pr(R = r_i | \pmb\theta_i),\; i=1 \ldots n`
    
    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    p = pi*(bitgamma(sample=sample, m=m, W=W, gamma=gamma)-1/m) + 1/m
    return p

def proba(m, pi, xi, r): #TODO proba
    """
    :DEPRECATED:
    """
    return None

def cmf(m, pi, gamma, W): #TODO: test cmf
    r"""Average cumulative probability of a specified CUB model
    with covariates for the feeling component.

    :math:`\Pr(R \geq r | \pmb\theta_i),\; r=1 \ldots m`
    
    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :return: the array of the cumulative probability distribution.
    :rtype: numpy array
    """
    return pmf(m, pi, gamma, W).cumsum()

def mean(m, pi, xi): #TODO mean
    return None

def var(m, pi, xi): #TODO var
    return None

def std(m, pi, xi): #TODO std
    return None

def skew(pi, xi): #TODO skew
    return None

def mean_diff(m, pi, xi): #TODO mean_diff
    return None
    
def median(m, pi, xi): #TODO median
    return None
    
def gini(m, pi, xi): #TODO gini
    return None
    
def laakso(m, pi, xi): #TODO laakso
    return None

def loglik(sample, m, pi, gamma, W):
    r"""Log-likelihood function of a CUB model with covariates for the feeling component

    Compute the log-likelihood function of a CUB model fitting ordinal data, with
    covariates for explaining the feeling component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :return: the log-likelihood
    :rtype: float
    """
    p = prob(sample, m, pi, gamma, W)
    l = np.sum(np.log(p))
    return l

def varcov(sample, m, pi, gamma, W):
    r"""Variance-covariance matrix of CUB models with covariates for the feeling component

    Compute the variance-covariance matrix of parameter estimates of a CUB model
    with covariates for the feeling component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :return: the log-likelihood
    :rtype: float
    """
    qi = 1/(m*prob(sample,m,pi,gamma,W))
    qistar = 1 - (1-pi)*qi
    qitilde = qistar*(1-qistar)
    fi = logis(W, gamma)
    fitilde = fi*(1-fi)
    ai = (sample-1) - (m-1)*(1-fi)
    g01 = (ai*qi*qistar)/pi
    hh = (m-1)*qistar*fitilde - (ai**2)*qitilde
    WW = addones(W)
    i11 = np.sum((1-qi)**2 / pi**2)
    i12 = g01.T @ WW
    i22 = WW.T @ hadprod(WW, hh)
    # Information matrix
    nparam = colsof(WW) + 1
    matinf = np.ndarray(shape=(nparam, nparam))
    matinf[:] = np.nan
    matinf[0,:] = np.concatenate([[i11], i12]).T

    varmat = np.ndarray(shape=(nparam, nparam))
    varmat[:] = np.nan
    for i in range(1, nparam):
        matinf[i,:] = np.concatenate([
            [i12[i-1]], i22[i-1,:]]).T
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        varmat = np.linalg.inv(matinf)
    return varmat

def init_gamma(sample, m, W):
    """
    Preliminary parameter estimates of a CUB model with covariates for feeling

    Compute preliminary parameter estimates for the feeling component of a CUB model 
    fitted to ordinal responses
    These estimates are set as initial values for parameters to start the E-M algorithm.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_0w.md#init_gammasample-m-w
    """
    WW = np.c_[np.ones(W.shape[0]), W]
    ni = np.log((m-sample+.5)/(sample-.5))
    gamma = np.linalg.inv(WW.T @ WW) @ (WW.T @ ni)
    return gamma

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, n, pi, gamma, W, seed=None):
    """
    Draw a random sample from CUB model

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_0w.md#drawm-n-pi-gamma-w
    """
    #np.random.seed(seed)
    assert n == W.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, pi=pi,
        gamma=gamma, W=W)
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
    theoric = pmf(m=m,pi=pi,gamma=gamma,W=W)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        [pi], gamma
    ))
    par_names = np.concatenate((
        ["pi"],
        ["constant"],
        W.columns
    ))
    sample = CUBsample(
        model="CUB(0W)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, W=W, diss=diss,
        theoric=theoric
    )
    return sample

###################################################################
# INFERENCE
###################################################################
def effe01(gamma, esterno01, m):
    """
    Auxiliary function for the log-likelihood estimation of CUB models

    Compute the opposite of the scalar function that is maximized when running 
    the E-M algorithm for CUB models with covariates for the feeling parameter.
    """
    ttau = esterno01[:,0]
    ordd = esterno01[:,1]
    covar = esterno01[:,2:]
    covar_gamma = covar @ gamma
    r = np.sum(
        ttau*(
            (ordd-1)*(covar_gamma)
            +
            (m-1)*np.log(1+np.exp(-covar_gamma))
        )
    )
    return r

def mle(sample, m, W,
    gen_pars=None,
    maxiter=500,
    tol=1e-4):
    """
    Main function for CUB models with covariates for the feeling component

    Function to estimate and validate a CUB model for given ordinal responses, with covariates for
    explaining the feeling component.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_0w.md#mlesample-m-w
    """
    # validate parameters
    #if not validate_pars(m=m, n=sample.size):
    #    pass
    # start datetime
    start = dt.datetime.now()
    # cast sample to numpy array
    sample = np.array(sample)
    # model preference choices
    #R = choices(m)
    # observed absolute frequecies
    f = freq(sample, m)
    # sample size
    n = sample.size
    #aver = np.mean(sample)
    # add a column of 1
    WW = addones(W)
    # number of covariates
    q = colsof(W)
    # initialize gamma parameter
    gammajj = init_gamma(sample=sample, m=m, W=W)
    # initialize (pi, xi)
    pijj, _ = cub.init_theta(f=f, m=m)
    # compute loglikelihood
    l = loglik(sample, m, pijj, gammajj, W)

    # start E-M algorithm
    niter = 1
    while niter < maxiter:
        lold = l
        vettn = bitgamma(sample=sample, m=m, W=W, gamma=gammajj)
        ttau = 1/(1+(1-pijj)/(m*pijj*vettn))
        #print(f"niter {niter} ***************")
        #print("vettn")
        #print(vettn)
        #print("ttau")
        #print(ttau)
        ################################# maximize w.r.t. gama  ########
        esterno01 = np.c_[ttau, sample, WW]
        optimgamma = minimize(
            effe01, x0=gammajj, args=(esterno01, m),
            method="Nelder-Mead"
            #method="BFGS"
        )
        #print(optimgamma)
        ################################################################
        gammajj = optimgamma.x #[0]
        #print(f"gama {gammajj}")
        pijj = np.sum(ttau)/n
        l = loglik(sample, m, pijj, gammajj, W)
        # compute delta-loglik
        deltal = abs(l-lold)
        # check tolerance
        if deltal <= tol:
            break
        else:
            lold = l
        niter += 1
    # end E-M algorithm
    pi = pijj
    gamma = gammajj
    #l = loglikjj
    # variance-covariance matrix
    varmat = varcov(sample, m, pi, gamma, W)
    end = dt.datetime.now()

    # Akaike Information Criterion
    AIC = aic(l=l, p=q+2)
    # Bayesian Information Criterion
    BIC = bic(l=l, p=q+2, n=n)

    #print(pi)
    #print(gamma)
    #print(niter)
    #print(l)
    #return None

    # standard errors
    stderrs = np.sqrt(np.diag(varmat))

    #print(stderrs)
    #return None
    # Wald statistics
    wald = np.concatenate([[pi], gamma])/stderrs
    #print(wald)
    #return None
    # p-value
    pval = 2*(sps.norm().sf(abs(wald)))
    # mean loglikelihood
    muloglik = l/n
    # loglik of null model (uniform)
    loglikuni = luni(n=n, m=m)
    # loglik of saturated model
    #logliksat = lsat(f=f, n=n)
    #TODO: TEST LOGLIK SAT FOR COVARIATES
    #      see https://stackoverflow.com/questions/77791392/proportion-of-each-unique-value-of-a-chosen-column-for-each-unique-combination-o#77791442
    #df = pd.merge(
    #    pd.DataFrame({"ord":sample}),
    #    W,
    #    left_index=True, right_index=True
    #)
    #df = pd.DataFrame({"ord":sample}).join(W)
    #cov = list(W.columns)
    #logliksatcov = np.sum(
    #    np.log(
    #    df.value_counts().div(
    #    df[cov].value_counts())))
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[W]
    #)
    # loglik of shiftet binomial
    # xibin = (m-sample.mean())/(m-1)
    # loglikbin = loglik(m, 1, xibin, f)
    # Explicative powers
    # Ebin = (loglikbin-loglikuni)/(logliksat-loglikuni)
    # Ecub = (l-loglikbin)/(logliksat-loglikuni)
    # Ecub0 = (l-loglikuni)/(logliksat-loglikuni)
    # deviance from saturated model
    #dev = 2*(logliksat-l)
    # ICOMP metrics
    #npars = q
    #trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + npars*np.log(trvarmat/npars) - np.log(np.linalg.det(varmat))
    # coefficient of correlation
    # rho = varmat[0,1]/np.sqrt(varmat[0,0]*varmat[1,1])
    theoric = pmf(m=m, pi=pi, gamma=gamma, W=W)
    diss = dissimilarity(f/n, theoric)
    gamma_names = np.concatenate([
        ["constant"],
        W.columns])
    estimates = np.concatenate((
        [pi], gamma
    ))
    est_names = np.concatenate((
        ["pi"], gamma_names
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        ["Feeling"],
        np.repeat(None, q)
    ))
    # compare with known (pi, xi)
    # if pi_gen is not None and xi_gen is not None:
    #     pass
    # results object
    res = CUBresCUB0W(
            model="CUB(0W)",
            m=m, n=n, niter=niter,
            maxiter=maxiter, tol=tol,
            estimates=estimates,
            est_names=est_names,
            e_types=e_types,
            stderrs=stderrs,
            pval=pval, wald=wald,
            loglike=l, muloglik=muloglik,
            loglikuni=loglikuni,
            #logliksat=logliksat,
            #logliksatcov=logliksatcov,
            # loglikbin=loglikbin,
            # Ebin=Ebin, Ecub=Ecub, Ecub0=Ecub0,
            theoric=theoric,
            #dev=dev,
            AIC=AIC, BIC=BIC,
            seconds=(end-start).total_seconds(),
            time_exe=start,
            # rho=rho,
            sample=sample, f=f,
            varmat=varmat,
            W=W,
            diss=diss,
            gen_pars=gen_pars
            # pi_gen=pi_gen, xi_gen=xi_gen
        )
    return res

class CUBresCUB0W(CUBres):
    """
    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_0w.md#cubrescub0w
    """

    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar", #options bar, scatter
        saveas=None
        ):
        """
        Plots relative frequencies of observed sample and estimated average probability mass.
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        title = "AVERAGE ESTIMATED PROBABILITY\n"
        title += f"{self.model} model    "
        title += f"$n={self.n}$\n"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        ax.set_title(title)

        R = choices(self.m)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability mass")

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

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        if ax is None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
            else:
                return fig, ax
        else:
            return ax

    def plot(self,
        #ci=.95,
        saveas=None,
        figsize=(7, 5)
        ):
        """
        plot CUB model fitted from a sample
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_ordinal(ax=ax)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
