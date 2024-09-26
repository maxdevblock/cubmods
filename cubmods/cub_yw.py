# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cubyw-module:

CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial) with covariates
for both feeling and uncertainty.

Description:
============
    This module contains methods and classes
    for CUB_YW model family.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    :math:`\Pr(R_i=r|\pmb{\theta} ; \pmb y_i ; \pmb w_i) = \pi_i \dbinom{m-1}{r-1}(1-\xi_i)^{r-1}\xi_i^{m-r}+\dfrac{1-\pi_i}{m},\; r=1,2,\ldots,m`

    :math:`\xi_i = \dfrac{1}{1+e^{-\pmb w_i \pmb\gamma}}`

    :math:`\pi_i = \dfrac{1}{1+e^{-\pmb y_i \pmb\beta}}`

Manual and Examples
==========================
    - `Models manual <manual.html#cub-models-with-covariates>`__


References:
===========
    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        piccolo2003moments
        d2005mixture
        piccolo2006observed
        iannario2010new
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

  
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
import scipy.stats as sps
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .general import (
    logis, bitgamma, freq, choices,
    hadprod, aic, bic, dissimilarity,
    luni,
    #lsat, lsatcov,
    addones, colsof
)
from .cub import (
    init_theta, pmf as pmf_cub
)
from .cub_0w import init_gamma, effe01
from .cub_y0 import effe10
from .smry import CUBres, CUBsample

def pmf(m, beta, gamma, Y, W):
    r"""Average probability distribution of a specified CUB model 
    with covariates for both feeling and uncertainty.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the vector of the probability distribution.
    :rtype: numpy array
    """
    p = pmfi(m, beta, gamma, Y, W)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, beta, gamma, Y, W):
    r"""Probability distribution for each subject of a specified CUB model 
    with covariates for both feeling and uncertainty.
    
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
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    pi_i = logis(Y, beta)
    xi_i = logis(W, gamma)
    n = W.shape[0]
    p = np.ndarray(shape=(n, m))
    for i in range(n):
        p[i,:] = pmf_cub(m=m, pi=pi_i[i],
            xi=xi_i[i])
    return p

def prob(m, sample, Y, W, beta, gamma):
    r"""Probability distribution of a CUB model with covariates for both feeling and uncertainty.

    Compute the probability distribution of a CUB model with covariates for both the feeling 
    and the uncertainty components.

    :math:`\Pr(R_i=r_i|\pmb\theta;\pmb T_i),\; i=1 \ldots n`

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    p = (
        logis(Y=Y, param=beta)*
        (bitgamma(sample=sample,m=m,
            W=W,gamma=gamma)-1/m)
        +1/m
    )
    return p

def draw(m, beta, gamma, Y, W,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUB model with covariates for
    both feeling and uncertainty.

    :param n: number of ordinal responses to be drawn
    :type n: int
    :param m: number of ordinal categories
    :type m: int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    #np.random.seed(seed)
    assert len(beta) == Y.shape[1]+1
    assert len(gamma) == W.shape[1]+1
    assert Y.shape[0] == W.shape[0]
    n = Y.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, beta=beta,
        gamma=gamma, W=W, Y=Y)
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
    theoric = pmf(m=m,beta=beta,
        gamma=gamma,W=W,Y=Y)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        beta, gamma
    ))
    par_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
    ))
    p_types = np.concatenate((
        np.repeat(["Uncertainty"], len(beta)),
        np.repeat(["Feeling"], len(gamma)),
    ))
    sample = CUBsample(
        model="CUB(YW)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        p_types=p_types,
        seed=seed, diss=diss,
        theoric=theoric,
        df=df, formula=formula
    )
    return sample

def loglik(m, sample, Y, W, beta, gamma):
    r"""Log-likelihood function of a CUB model with covariates for both feeling and uncertainty.

    Compute the log-likelihood function of a CUB model fitting ordinal data
    with covariates for explaining both the feeling and the uncertainty components.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the log-likelihood value
    :rtype: float
    """
    p = prob(m, sample, Y, W, beta, gamma)
    l = np.sum(np.log(p))
    return l

def varcov(m, sample, Y, W, beta, gamma):
    r"""Variance-covariance matrix of a CUB model with covariates for both uncertainty and feeling.

    Compute the variance-covariance matrix of parameter estimates of a CUB model with covariates for
    both the uncertainty and the feeling components.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the variance-covariance matrix of the CUB model
    :rtype: numpy ndarray
    """
    qi = 1/(m*prob(m=m, sample=sample,
        Y=Y, W=W, beta=beta, gamma=gamma))
    ei = logis(Y, beta)
    eitilde = ei*(1-ei)
    qistar = 1-(1-ei)*qi
    qitilde = qistar*(1-qistar)
    fi = logis(W, gamma)
    fitilde = fi*(1-fi)
    ai = (sample-1)-(m-1)*(1-fi)
    ff = eitilde-qitilde
    gg = ai*qitilde
    hh = (m-1)*qistar*fitilde-(ai**2)*qitilde
    YY = addones(Y)
    WW = addones(W)
    i11 = YY.T @ hadprod(YY, ff)
    i12 = YY.T @ hadprod(WW, gg)
    i22 = WW.T @ hadprod(WW, hh)
    npar = beta.size + gamma.size
    infmat = np.ndarray(shape=(npar,npar))
    for i in range(beta.size):
        infmat[i,:] = np.concatenate((
            i11[i,:], i12[i,:]
        )).T
    for i in range(beta.size, npar):
        infmat[i,:] = np.concatenate((
            i12.T[i-beta.size,:],
            i22[i-beta.size,:]
        )).T
    varmat = np.ndarray(shape=(npar,npar))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    return varmat

def mle(sample, m, Y, W,
    df, formula,
    ass_pars=None,
    maxiter=500,
    tol=1e-4):
    r"""Main function for CUB models with covariates for both the uncertainty and the feeling components.

    Estimate and validate a CUB model for given ordinal responses, with covariates for explaining both the
    feeling and the uncertainty components by means of logistic transform.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param Y: dataframe of covariates for explaining the uncertainty component
    :type Y: pandas dataframe
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
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
    :return: an instance of ``CUBresCUBYW`` (see the Class for details)
    :rtype: object
    """
    # start datetime
    start = dt.datetime.now()
    # cast sample to numpy array
    sample = np.array(sample)
    # model ordinal categories
    #R = choices(m)
    # observed absolute frequecies
    f = freq(sample, m)
    # sample size
    n = sample.size
    #aver = np.mean(sample)
    # add a column of 1
    YY = addones(Y)
    WW = addones(W)
    # number of covariates
    q = colsof(W)
    p = colsof(Y)
    # init
    pi, _ = init_theta(f=f, m=m)
    beta0 = np.log(pi/(1-pi))
    betajj = np.concatenate((
        [beta0],
        np.repeat(.1, p)
    ))
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    gammajj = init_gamma(
        #sample=rank, #TODO: in R is factor(sample, ordered=T)
        sample=sample,
        m=m, W=W)
    l = loglik(m=m, sample=sample,
        Y=Y, W=W,
        beta=betajj, gamma=gammajj)
    # start EM
    niter = 1
    while niter < maxiter:
        lold = l
        vettn = bitgamma(
            sample=sample,
            # sample=rank,
            m=m,
            W=W, gamma=gammajj
        )#[sample-1]
        aai = -1 + 1/logis(Y=Y, param=betajj)
        ttau = 1/(1 + aai/(m*vettn))
        esterno10 = np.c_[ttau, YY]
        esterno01 = np.c_[ttau, sample, WW]
        betaoptim = minimize(
            effe10, x0=betajj, args=(esterno10),
            #method="Nelder-Mead"
            #method="BFGS"
        )
        gamaoptim = minimize(
            effe01, x0=gammajj, args=(esterno01, m),
            #method="Nelder-Mead"
            #method="BFGS"
        )
        betajj = betaoptim.x
        gammajj = gamaoptim.x
        l = loglik(m, sample, Y, W, betajj, gammajj)
        # compute delta-loglik
        deltal = abs(l-lold)
        # check tolerance
        if deltal <= tol:
            break
        else:
            lold = l
        niter += 1
    # end E-M algorithm
    beta = betajj
    gamma = gammajj
    #l = loglikjj
    # variance-covariance matrix
    varmat = varcov(m, sample, Y, W, beta, gamma)
    stderrs = np.sqrt(np.diag(varmat))
    wald = np.concatenate((beta,gamma))/stderrs
    # p-value
    pval = 2*(sps.norm().sf(abs(wald)))

    muloglik = l/n
    AIC = aic(l=l, p=wald.size)
    BIC = bic(l=l, p=wald.size, n=n)
    theoric = pmf(m, beta, gamma, Y, W)
    diss = dissimilarity(f/n, theoric)
    loglikuni = luni(m=m, n=n)
    #logliksat = lsat(f=f, n=n)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[Y, W]
    #)
    #dev = 2*(logliksat-l)

    beta_names = np.concatenate([
        ["constant"],
        Y.columns])
    gamma_names = np.concatenate([
        ["constant"],
        W.columns])
    estimates = np.concatenate((
        beta, gamma
    ))
    est_names = np.concatenate((
        beta_names, gamma_names
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        np.repeat(None, p),
        ["Feeling"],
        np.repeat(None, q)
    ))

    end = dt.datetime.now()

    return CUBresCUBYW(
        model="CUB(YW)",
        m=m, n=n, niter=niter,
        maxiter=maxiter, tol=tol,
        estimates=estimates,
        est_names=est_names,
        e_types=e_types,
        theoric=theoric,
        stderrs=stderrs, wald=wald,
        pval=pval, loglike=l,
        muloglik=muloglik,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        AIC=AIC, BIC=BIC,
        seconds=(end-start).total_seconds(),
        time_exe=start,
        sample=sample, f=f,
        varmat=varmat, df=df,
        formula=formula,
        diss=diss,
        ass_pars=ass_pars
        #dev=dev
    )

class CUBresCUBYW(CUBres):
    r""""Object returned by ``.mle()`` function.
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
        
        title = "AVERAGE ESTIMATED PROBABILITY\n"
        title += f"{self.model} model    "
        title += f"$n={self.n}$\n"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
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
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
