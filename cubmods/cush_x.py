# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cushx-module:

CUB models in Python.
Module for CUSH (Combination of Uniform
and Shelter effect) with covariates.

Description:
============
    This module contains methods and classes
    for CUSH model family.

Manual, Examples and References:
================================
    - `Models manual <manual.html#cush-with-covariates>`__

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
#import pickle
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
from statsmodels.tools.numdiff import approx_hess
import matplotlib.pyplot as plt
from .general import (
    logis, freq, dissimilarity,
    aic, bic, 
    #lsat, 
    luni, choices,
    #lsatcov, 
    addones, colsof,
)
from .cush import pmf as pmf_cush
from .smry import CUBres, CUBsample

def pmf(m, sh, omega, X):
    r"""Average probability distribution of a specified CUSH model with covariates.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :return: the probability distribution
    :rtype: array
    """
    p = pmfi(m, sh, omega, X)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, sh, omega, X):
    r"""Probability distribution for each subject of a specified CUSH model with covariates

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    delta = logis(X, omega)
    #print(delta)
    n = X.shape[0]
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = pmf_cush(m=m, sh=sh, 
            delta=delta[i])
    return p

def prob(m, sample, X, omega, sh):
    r"""Probability distribution of a specified CUSH model with covariates.

    :math:`\Pr(R_i=r_i|\pmb\theta;\pmb T_i),\;i = 1 \ldots n`

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :return: the probability array :math:`\Pr(R = r | \pmb\theta)` for observed responses
    :rtype: float
    """
    delta = logis(X, omega)
    D = (sample==sh).astype(int)
    p = delta*(D-1/m)+1/m
    return p

def draw(m, sh, omega, X,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUSH model with covariates

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` (see `here <cubmods.html#cubmods.smry.CUBsample>`__) containing ordinal responses drawn from the specified model
    """
    n = X.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    R = choices(m)
    p = pmfi(m, sh, omega, X)
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
        X.columns
    ))
    p_types = np.repeat("Shelter", len(omega))
    
    return CUBsample(
        model="CUSH(X)",
        m=m, sh=sh,
        pars=omega,
        par_names=par_names,
        p_types=p_types,
        theoric=theoric,
        diss=diss,
        df=df, formula=formula,
        rv=rv.astype(int),
        seed=seed
    )

def loglik(m, sample, X, omega, sh):
    r"""Log-likelihood function for CUSH models with covariates.

    Compute the log-likelihood function for CUSH models with covariates 
    to explain the shelter effect.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param omega: array :math:`\pmb \omega` of parameters for the shelter effect, whose length equals 
        ``X.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega: array
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :return: the log-likelihood value
    :rtype: float
    """
    p = prob(m=m, sample=sample, X=X,
        omega=omega, sh=sh)
    l = np.sum(np.log(p))
    return l

def effe(pars, esterno, m, sh):
    r"""Auxiliary function for the log-likelihood estimation of CUSH models with covariates

    Compute the opposite of the loglikelihood function for CUSH models
    with covariates to explain the shelter effect.
    It is called as an argument for "optim" within ``.mle()`` function
    as the function to minimize.

    :param pars: array of the initial parameters estimates
    :type pars: array
    :param esterno: matrix binding together the vector of ordinal data and the matrix ``XX`` of explanatory
        variables whose first column is a column of ones needed to consider an intercept term
    :type esterno: ndarray
    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    """
    sample = esterno[:,0]
    X = esterno[:,2:] # no 1
    l = loglik(m=m, sample=sample, X=X,
        omega=pars, sh=sh)
    return -l

def mle(m, sample, X, sh,
    df, formula, ass_pars=None,
    maxiter=None, tol=None #for GEM compatibility
    ):
    r"""Main function for CUSH models with covariates.

    Estimate and validate a CUSH model for ordinal responses, with covariates
    to explain the shelter effect.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param X: dataframe of covariates for explaining the shelter effect
    :type X: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param ass_pars: dictionary of hypothesized parameters, defaults to None
    :type ass_pars: dictionary, optional
    :param maxiter: default to None; ensure compatibility with ``gem.from_formula()``
    :type maxiter: None
    :param tol: default to None; ensure compatibility with ``gem.from_formula()``
    :type tol: None
    :return: an instance of ``CUBresCUSHX`` (see the Class for details)
    :rtype: object
    """
    _, _ = maxiter, tol
    start = dt.datetime.now()
    n = sample.size
    f = freq(sample=sample, m=m)
    fc = f[sh-1]/n
    delta = max([.01, (m*fc-1)/(m-1)])
    XX = addones(X)
    x = colsof(X)
    om0 = np.log(delta/(1-delta))
    omi = np.concatenate((
        [om0], np.repeat(.1, x)
    ))
    esterno = np.c_[sample, XX]
    optim = minimize(
        effe, x0=omi,
        args=(esterno, m, sh),
        method="BFGS"
        #method="dogleg"
    )
    omega = optim.x
    l = loglik(m=m, sample=sample, X=X,
        omega=omega, sh=sh)
    muloglik = l/n
    infmat = approx_hess(omega, effe,
        args=(esterno, m, sh))
    varmat = np.ndarray(shape=(omega.size,omega.size))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    stderrs = np.sqrt(np.diag(varmat))
    wald = omega/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    
    AIC = aic(l=l, p=omega.size)
    BIC = bic(l=l, p=omega.size, n=n)
    loglikuni = luni(m=m, n=n)
    #logliksat = lsat(f=f, n=n)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[X]
    #)
    #dev = 2*(logliksat-l)
    theoric = pmf(m=m, omega=omega, X=X, sh=sh)
    diss = dissimilarity(f/n, theoric)
    omega_names = np.concatenate([
        ["constant"],
        X.columns])
    e_types = np.concatenate((
        ["Shelter effect"],
        [None for _ in X.columns]
    ))
    end = dt.datetime.now()
    return CUBresCUSHX(
        model="CUSH(X)",
        m=m, n=n, sh=sh, estimates=omega,
        est_names=omega_names,
        e_types=e_types,
        stderrs=stderrs, pval=pval,
        theoric=theoric,
        wald=wald, loglike=l,
        muloglik=muloglik,
        loglikuni=loglikuni,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        #dev=dev,
        AIC=AIC, BIC=BIC,
        sample=sample, f=f, varmat=varmat,
        diss=diss, df=df, formula=formula,
        seconds=(end-start).total_seconds(),
        time_exe=start, ass_pars=ass_pars
    )

class CUBresCUSHX(CUBres):
    r"""Object returned by ``.mle()`` function.
    See `here <cubmods.html#cubmods.smry.CUBres>`__ the Base for details.
    """
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
        
        if self.ass_pars is not None:
            ddf = self.as_dataframe()
            Xcols = ddf[
                (ddf.component=="Shelter effect")
                &
                (ddf.parameter!="constant")
            ].parameter.values
            ass_p = pmf(
                m=self.m, sh=self.sh,
                omega=self.ass_pars["omega"],
                X=self.df[Xcols]
            )
            ax.stem(R, ass_p, linefmt="--r",
                markerfmt="none", label="assumed")


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

        :param figsize: tuple of ``(length, height)`` for the figure
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
