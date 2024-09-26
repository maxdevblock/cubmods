# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cush2xx-module:

CUB models in Python.
Module for CUSH2 (Combination of Uniform
and 2 Shelter Choices) with covariates.

Description:
============
    This module contains methods and classes
    for CUSH2 model family with covariates for
    both shelter choices.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.
    The CUSH2 family has been defined and implemented
    by Massimo Pierini (2024) in the thesis
    *Modelli della classe CUB in Python*.

    :math:`\Pr(R_i=r|\pmb \theta;\pmb x_{1i};\pmb x_{2i}) = \delta_{1i} D_r^{(c_1)} + \delta_{2i} D_r^{(c_2)} + (1 - \delta_{1i} - \delta_{2i})/m,\; r=1,2,\ldots,m`

    :math:`\delta_{1i} = \dfrac{1}{1+\exp\{ - \pmb x_{1i} \pmb \omega_1 \}}`

    :math:`\delta_{2i} = \dfrac{1}{1+\exp\{ - \pmb x_{2i} \pmb \omega_2 \}}`

Manual, Examples and References:
===============================
    - `Models manual <manual.html#cush2-with-covariates>`__

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
    logis, colsof, choices, freq,
    aic, bic, 
    #lsat, 
    luni, dissimilarity,
    #lsatcov
)
from .cush2 import pmf as pmf_cush2
from .smry import CUBres, CUBsample

def pmfi(m, sh1, sh2,
    omega1, omega2,
    X1, X2):
    r"""Probability distribution for each subject of a specified CUSH2 model with covariates
    for both shelter choices.

    Auxiliary function of ``.draw()``.

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param omega1: array :math:`\pmb \omega_1` of parameters for the 1st shelter effect, whose length equals 
        ``X1.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega1: array
    :param omega2: array :math:`\pmb \omega_2` of parameters for the 2nd shelter effect, whose length equals 
        ``X2.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega2: array
    :param X1: dataframe of covariates for explaining the 1st shelter effect
    :type X1: DataFrame
    :param X2: dataframe of covariates for explaining the 2nd shelter effect
    :type X2: DataFrame
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    d1 = logis(X1, omega1)
    d2 = logis(X2, omega2)
    n = X1.shape[0]
    p_i = np.ndarray(shape=(n,m))
    for i in range(n):
        p_i[i] = pmf_cush2(m=m, c1=sh1,
            c2=sh2, d1=d1[i], d2=d2[i])
    return p_i

def pmf(m, sh1, sh2,
    omega1, omega2,
    X1, X2):
    r"""Average probability distribution of a specified CUSH2 model with covariates
    for both shelter choices.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param omega1: array :math:`\pmb \omega_1` of parameters for the 1st shelter effect, whose length equals 
        ``X1.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega1: array
    :param omega2: array :math:`\pmb \omega_2` of parameters for the 2nd shelter effect, whose length equals 
        ``X2.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega2: array
    :param X1: dataframe of covariates for explaining the 1st shelter effect
    :type X1: DataFrame
    :param X2: dataframe of covariates for explaining the 2nd shelter effect
    :type X2: DataFrame
    :return: the average probability distribution
    :rtype: array
    """
    p_i = pmfi(m, sh1, sh2, omega1, omega2,
        X1, X2)
    p = p_i.mean(axis=0)
    return p

def draw(m, sh1, sh2, omega1, omega2, X1,
    X2, df, formula, seed=None):
    r"""Draw a random sample from a specified CUSH2 model,
    with covariates for both shelter choices.

    :param m: number of ordinal categories
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param omega1: array :math:`\pmb \omega_1` of parameters for the 1st shelter effect, whose length equals 
        ``X1.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega1: array
    :param omega2: array :math:`\pmb \omega_2` of parameters for the 2nd shelter effect, whose length equals 
        ``X2.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega2: array
    :param X1: dataframe of covariates for explaining the 1st shelter effect
    :type X1: DataFrame
    :param X2: dataframe of covariates for explaining the 2nd shelter effect
    :type X2: DataFrame
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    n = X1.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m, sh1, sh2, omega1,
        omega2, X1, X2)
    for i in range(n):
        if seed is not None:
            np.random.seed(seed*i)
        rv[i] = np.random.choice(
            choices(m),
            size=1,
            replace=True,
            p=theoric_i[i]
        )
    f = freq(m=m, sample=rv)
    theoric = theoric_i.mean(axis=0)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        omega1, omega2
    ))
    par_names = np.concatenate((
        ["constant"], X1.columns,
        ["constant"], X2.columns
    ))
    p_types = np.concatenate((
        np.repeat(["Shelter1"], len(omega1)),
        np.repeat(["Shelter2"], len(omega2)),
    ))
    return CUBsample(
        model="CUSH2(X1,X2)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        p_types=p_types,
        seed=seed, diss=diss,
        theoric=theoric, sh=[sh1, sh2],
        df=df, formula=formula
    )

def loglik(sample, m, sh1, sh2,
    omega1, omega2,
    X1, X2):
    r"""Log-likelihood function for a CUSH2 model with covariates
    for both shelter choices.

    Compute the log-likelihood function for a CUSH2 model 
    with covariates for both shelter choices,
    for the given ordinal responses.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param omega1: array :math:`\pmb \omega_1` of parameters for the 1st shelter effect, whose length equals 
        ``X1.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega1: array
    :param omega2: array :math:`\pmb \omega_2` of parameters for the 2nd shelter effect, whose length equals 
        ``X2.columns.size+1`` to include an intercept term in the model (first entry)
    :type omega2: array
    :param X1: dataframe of covariates for explaining the 1st shelter effect
    :type X1: DataFrame
    :param X2: dataframe of covariates for explaining the 2nd shelter effect
    :type X2: DataFrame
    :return: the log-likehood value
    :rtype: float
    """
    delta1 = logis(X1, omega1)
    delta2 = logis(X2, omega2)
    D1 = (sample==sh1).astype(int)
    D2 = (sample==sh2).astype(int)
    l = np.sum(np.log(
        delta1*D1 + delta2*D2 +
        (1-delta1-delta2)/m
    ))
    return l

def effe(pars, sample, m, sh1, sh2, X1, X2):
    r"""Auxiliary function for the log-likelihood estimation of CUSH2 models.

    Compute the opposite of the scalar function that is maximized when running
    the E-M algorithm for CUSH2 models with covariates for both shelter choices.

    :param pars: array of parameters
    :type pars: array
    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param X1: dataframe of covariates for explaining the 1st shelter effect
    :type X1: DataFrame
    :param X2: dataframe of covariates for explaining the 2nd shelter effect
    :type X2: DataFrame
    """
    w1 = colsof(X1)+1
    omega1 = pars[:w1]
    omega2 = pars[w1:]
    l = loglik(sample, m, sh1, sh2,
        omega1, omega2,
        X1, X2)
    return -l

def mle(sample, m, sh1, sh2,
    X1, X2, df, formula, ass_pars=None):
    r"""Main function for CUSH2 models with covariates for both shelter choices.

    Estimate and validate a CUSH2 model for given ordinal responses, with covariates for both shelter choices.
    
    :param sample: array of ordinal responses
    :type sample: array of int
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param X1: dataframe of covariates for explaining the 1st shelter effect
    :type X1: DataFrame
    :param X2: dataframe of covariates for explaining the 2nd shelter effect
    :type X2: DataFrame
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param ass_pars: dictionary of hypothesized parameters, defaults to None
    :type ass_pars: dictionary, optional
    :return: an instance of ``CUBresCUSH2XX`` (see the Class for details)
    :rtype: object
    """
    start = dt.datetime.now()
    w1 = colsof(X1)
    w2 = colsof(X2)
    n = sample.size
    f = freq(m=m, sample=sample)
    fc1 = (sample==sh1).sum()/n
    fc2 = (sample==sh2).sum()/n
    delta1_0 = max([
        .01, (fc1*(m-1)+fc2-1)/(m-2)])
    om1_0 = np.log(delta1_0/(1-delta1_0))
    om1 = np.concatenate((
        [om1_0], np.repeat(0, w1)
    ))
    delta2_0 = max([
        .01, (fc2*(m-1)+fc1-1)/(m-2)])
    om2_0 = np.log(delta2_0/(1-delta2_0))
    om2 = np.concatenate((
        [om2_0], np.repeat(0, w2)
    ))
    pars = np.concatenate((om1, om2))
    optim = minimize(
        effe, x0=pars,
        args=(sample, m, sh1, sh2, X1, X2)
    )
    estimates = optim.x
    omega1 = estimates[:(w1+1)]
    omega2 = estimates[(w1+1):]
    est_names = np.concatenate((
        ["constant"],
        [x for x in X1.columns],
        ["constant"],
        [x for x in X2.columns],
    ))
    e_types = np.concatenate((
        ["Shelter effect 1"],
        [None for _ in X1.columns],
        ["Shelter effect 2"],
        [None for _ in X2.columns],
    ))
    
    infmat = approx_hess(estimates, effe,
        args=(sample, m, sh1, sh2, X1, X2))
    varmat = np.ndarray(shape=(
        estimates.size,estimates.size))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    stderrs = np.sqrt(np.diag(varmat))
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    l = loglik(m=m, sample=sample,
        omega1=omega1, omega2=omega2,
        sh1=sh1, sh2=sh2, X1=X1, X2=X2)
    #logliksat = lsat(n=n, f=f)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[X1, X2]
    #)
    loglikuni = luni(m=m, n=n)
    #dev = 2*(logliksat-l)
    theoric = pmf(m=m, sh1=sh1, sh2=sh2,
        omega1=omega1, omega2=omega2,
        X1=X1, X2=X2)
    diss = dissimilarity(f/n, theoric)
    muloglik = l/n
    AIC = aic(l=l, p=estimates.size)
    BIC = bic(l=l, p=estimates.size, n=n)
    end = dt.datetime.now()
    
    return CUBresCUSH2XX(
        model="CUSH2(X1,X2)",
        m=m, n=n, sh=np.array([sh1, sh2]),
        estimates=estimates,
        est_names=est_names,
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

class CUBresCUSH2XX(CUBres):
    """Object returned by ``.mle()`` function.
    See the Base for details.
    """
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        r"""Plots relative average frequencies of observed sample, estimated average probability distribution and,
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