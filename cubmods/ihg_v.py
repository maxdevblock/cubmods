# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _ihgv-module:

CUB models in Python.
Module for IHG (Inverse HyperGeometric) with covariates.

Description:
============
    This module contains methods and classes
    for IHG model family with covariates.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    .. math::
        \left\{
        \begin{array}{l}
            \Pr(R_i=1|\pmb\theta;\pmb v_i) = \theta_i
            \\
            \Pr(R_i=r+1|\pmb\theta;\pmb v_i) = \Pr(R_i=r|\pmb\theta ; \pmb v_i)(1-\theta_i)\dfrac{m-r}{m-1-r(1-\theta_i)},\; r= 1, \ldots, m-1
        \end{array}
        \right.

    .. math::
        \theta_i = \dfrac{1}{1 + \exp\{ - \pmb \nu_i \pmb v \}}
    
Manual and Examples
==========================
    - `Models manual <manual.html#ihg-with-covariates>`__

References:
===========
    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        d2003modelling
        d2005moment
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
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_hess
from .general import (
    logis, freq, choices, aic, bic,
    #lsat, 
    luni, dissimilarity,
    #lsatcov, 
    #addones, 
    colsof,
)
from .ihg import pmf as pmf_ihg
from .smry import CUBres, CUBsample

def pmfi(m, V, nu):
    r"""Probability distribution for each subject of a specified IHG model with covariates

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param nu: array :math:`\pmb \nu` of parameters for :math:`\theta`, whose length equals 
        ``V.columns.size+1`` to include an intercept term in the model (first entry)
    :type nu: array
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    n = V.shape[0]
    p_i = np.ndarray(shape=(n,m))
    theta = logis(V, nu)
    for i in range(n):
        p_i[i] = pmf_ihg(m=m, theta=theta[i])
    return p_i

def pmf(m, V, nu):
    r"""Average probability distribution of a specified IHG model with covariates.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param nu: array :math:`\pmb \nu` of parameters for :math:`\theta`, whose length equals 
        ``V.columns.size+1`` to include an intercept term in the model (first entry)
    :type nu: array
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :return: the probability distribution
    :rtype: array
    """
    p = pmfi(m, V, nu).mean(axis=0)
    return p

def draw(m, nu, V, 
    df, formula, seed=None):
    r"""Draw a random sample from a specified IHG model with covariates

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param nu: array :math:`\pmb \nu` of parameters for :math:`\theta`, whose length equals 
        ``V.columns.size+1`` to include an intercept term in the model (first entry)
    :type nu: array
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    n = V.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    R = choices(m)
    p = pmfi(m, V, nu)
    rv = np.repeat(np.nan, n)
    for i in range(n):
        if seed is not None:
            np.random.seed(seed*i)
        rv[i] = np.random.choice(
            R,
            size=1, replace=True,
            p=p[i]
        )
    theoric = p.mean(axis=0)
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    par_names = np.concatenate((
        ["constant"],
        V.columns
    ))
    
    return CUBsample(
        model="IHG(V)",
        m=m,
        pars=nu,
        par_names=par_names,
        theoric=theoric,
        diss=diss,
        df=df, formula=formula,
        rv=rv.astype(int),
        seed=seed
    )

def prob(m, sample, V, nu):
    r"""Probability distribution of a IHG model with covariates
    given an observed sample.

    Compute the probability distribution of a IHG model with covariates, 
    given an observed sample.
    
    :math:`\Pr(R_i=r_i|\pmb\theta;\pmb T_i),\; i=1 \ldots n`

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param nu: array :math:`\pmb \nu` of parameters for :math:`\theta`, whose length equals 
        ``V.columns.size+1`` to include an intercept term in the model (first entry)
    :type nu: array
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    n = sample.size
    theta = logis(V, nu)
    p = np.repeat(np.nan, n)
    for i in range(n):
        prob = pmf_ihg(m=m, theta=theta[i])
        p[i] = prob[sample[i]-1]
    return p

def loglik(m, sample, V, nu):
    r"""Log-likelihood function for IHG models with covariates.

    Compute the log-likelihood function for CUSH models with covariates 
    to explain the shelter effect.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param nu: array :math:`\pmb \nu` of parameters for :math:`\theta`, whose length equals 
        ``V.columns.size+1`` to include an intercept term in the model (first entry)
    :type nu: array
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :return: the log-likelihood value
    :rtype: float
    """
    p = prob(m, sample, V, nu)
    l = np.sum(np.log(p))
    return l

def effe(nu, m, sample, V):
    r"""Auxiliary function for the log-likelihood estimation of IHG models with covariates

    Compute the opposite of the loglikelihood function for IHG models
    with covariates.
    It is called as an argument for "optim" within ``.mle()`` function
    as the function to minimize.

    :param nu: initial parameter estimate
    :type nu: float
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    """
    l = loglik(m, sample, V, nu)
    return -l

def init_theta(m, f):
    r"""Preliminary estimators for IHG models without covariates.

    Computes preliminary parameter estimates of a IHG model without covariates for given ordinal
    responses. These preliminary estimators are used within the package code to start the E-M algorithm.

    :param f: array of the absolute frequencies of given ordinal responses
    :type f: array of int
    :param m: number of ordinal categories
    :type m: int
    :return: the array of :math:`\pmb\nu^{(0)}`
    """
    R = choices(m)
    aver = np.sum(f*R)/np.sum(f)
    est = (m-aver)/(1+(m-2)*aver)
    return est

def mle(m, sample, V,
    df, formula, gen_pars=None):
    r"""Main function for IHG models with covariates.

    Estimate and validate a IHG model for ordinal responses, with covariates.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param V: dataframe of covariates for explaining the parameter :math:`\theta`
    :type V: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param gen_pars: dictionary of hypothesized parameters, defaults to None
    :type gen_pars: dictionary, optional
    :return: an instance of ``CUBresIHGV`` (see the Class for details)
    :rtype: object
    """
    start = dt.datetime.now()
    f = freq(m=m, sample=sample)
    n = sample.size
    theta0 = init_theta(m, f)
    #VV = addones(V)
    v = colsof(V)
    nu0 = np.log(theta0/(1-theta0))
    nuini = np.concatenate((
        [nu0], np.repeat(.1, v)
    ))
    optim = minimize(
        effe, x0=nuini,
        args=(m, sample, V),
        #method="Nelder-Mead"
    )
    nu = optim.x
    l = loglik(m, sample, V, nu)
    infmat = approx_hess(nu, effe,
        args=(m, sample, V))
    varmat = np.ndarray(shape=(nu.size,nu.size))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.array(nu)
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    l = loglik(m=m, sample=sample, nu=nu,
        V=V)
    #logliksat = lsat(n=n, f=f)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[V]
    #)
    loglikuni = luni(m=m, n=n)
    muloglik = l/n
    #dev = 2*(logliksat-l)
    AIC = aic(l=l, p=estimates.size)
    BIC = bic(l=l, p=estimates.size, n=n)
    theoric = pmf(m=m, nu=nu, V=V)
    diss = dissimilarity(f/n, theoric)
    est_names = np.concatenate((
        ["constant"],
        V.columns
    ))
    #print(est_names.shape)
    e_types = np.concatenate((
        ["Theta"],
        [None for _ in V.columns]
    ))
    
    end = dt.datetime.now()

    return CUBresIHGV(
        model="IHG(V)",
        m=m, n=n,
        theoric=theoric,
        e_types=e_types,
        est_names=est_names,
        estimates=estimates,
        stderrs=stderrs,
        pval=pval,
        wald=wald, loglike=l,
        muloglik=muloglik,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        #dev=dev,
        AIC=AIC, BIC=BIC,
        diss=diss, sample=sample,
        f=f, varmat=varmat,
        seconds=(end-start).total_seconds(),
        time_exe=start, gen_pars=gen_pars,
        df=df, formula=formula
    )

class CUBresIHGV(CUBres):
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        r"""Plots avreage relative frequencies of observed sample, estimated 
        average probability distribution and,
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