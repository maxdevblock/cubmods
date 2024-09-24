# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cush0-module:

CUB models in Python.
Module for CUSH (Combination of Uniform
and Shelter effect).

Description:
============
    This module contains methods and classes
    for CUSH model family.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    :math:`\Pr(R=r|\pmb \theta) = \delta D_r^{(c)} + (1 - \delta)/m,\; r=1,2,\ldots,m`

Manual and Examples
==========================
    - `Models manual <manual.html#cush-without-covariates>`__

References:
===========
    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        capecchi2017dealing
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

  
List of TODOs:
==============
  - TODO: check and fix gini & laakso

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
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, dissimilarity,
    #chisquared,
    lsat, luni, aic, bic,
    NoShelterError
)
#from . import cub
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def pmf(m, sh, delta):
    r"""Probability distribution of a specified CUSH model.

    :math:`\Pr(R = r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: the probability distribution
    :rtype: array
    """
    R = choices(m=m)
    s = (R==sh).astype(int)
    p = delta*(s-1/m)+1/m
    return p

def loglik(sample, m, sh, delta):
    r"""Log-likelihood function for a CUSH model without covariates

    Compute the log-likelihood function for a CUSH model 
    without covariate for the given ordinal responses.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: the log-likehood value
    :rtype: float
    """
    n = sample.size
    f = freq(sample=sample, m=m)
    fc = f[sh-1]/n
    l = n*((1-fc)*np.log(1-delta)
        +fc*np.log(1+(m-1)*delta)
        -np.log(m)
    )
    return l

def mean(m, sh, delta):
    r"""Expected value of a specified CUSH model.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: the expected value of the model
    :rtype: float
    """
    mu = delta*sh+(1-delta)*(m+1)/2
    return mu

def var(m, sh, delta):
    r"""Variance of a specified CUSH model.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: the variance of the model
    :rtype: float
    """
    va = (1-delta)*(delta*(sh-(m+1)/2)**2+(m**2-1)/12)
    return va

def gini(delta):
    r"""The Gini index of a specified CUSH model.

    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: the Gini index of the model
    :rtype: float
    """
    return 1-delta**2

def laakso(m, delta):
    r"""The Laakso index of a specified CUSH model.

    :param m: number of ordinal categories
    :type m: int
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
    :return: the Laakso index of the model
    :rtype: float
    """
    l = (1-delta**2)/(1+(m-1)*delta**2)
    return l

def LRT(m, fc, n):
    r"""Likelihood Ratio Test between the CUSH model and
    the null model.

    :param m: number of ordinal categories
    :type m: int
    :param fc: relative frequency of the shelter category
    :type fc: float
    :param n: number of observations
    :type n: int
    :return: the value of the LRT
    :rtype: float
    """
    a = fc*np.log(fc)
    b = (1-fc)*np.log((1-fc)/(m-1))
    c = np.log(m)
    return 2*n*(a+b+c)

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, sh, delta, n,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUSH model.

    :param m: number of ordinal categories
    :type m: int
    :param sh: Category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param delta: shelter choice parameter :math:`\delta`
    :type delta: float
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
    if sh is None:
        raise NoShelterError(model="cush")
    theoric = pmf(m=m, sh=sh, delta=delta)
    np.random.seed(seed)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=theoric
        )
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    pars = np.array([delta])
    par_names = np.array(["delta"])
    sample = CUBsample(
        model="CUSH",
        rv=rv, m=m,
        sh=sh, pars=pars,
        par_names=par_names,
        seed=seed, theoric=theoric,
        diss=diss, df=df,
        formula=formula
    )
    return sample

###################################################################
# INFERENCE
###################################################################

def mle(sample, m, sh, df, formula,
    ass_pars=None,
    maxiter=None, tol=None #for GEM compatibility
    ):
    r"""Main function for CUSH model without covariates.

    Estimate and validate a CUSH model for given ordinal responses, without covariates.

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
    :param maxiter: default to None; ensure compatibility with ``gem.from_formula()``
    :type maxiter: None
    :param tol: default to None; ensure compatibility with ``gem.from_formula()``
    :type tol: None
    :return: an instance of ``CUBresCUSH`` (see the Class for details)
    :rtype: object
    """
    _, _ = maxiter, tol
    if sh is None:
        raise NoShelterError(model="cush")
    start = dt.datetime.now()
    f = freq(sample=sample, m=m)
    n = sample.size
    #aver = np.mean(sample)
    fc = f[sh-1]/n
    deltaest = np.max([.01, (m*fc-1)/(m-1)])
    esdelta = np.sqrt(
        (1-deltaest)*(1+(m-1)*deltaest)/
        (n*(m-1))
    )
    
    varmat = esdelta**2
    end = dt.datetime.now()
    wald = deltaest/esdelta
    pval = 2*(sps.norm().sf(abs(wald)))
    l = loglik(
        sample=sample, m=m,
        sh=sh, delta=deltaest
    )
    AIC = aic(l=l, p=1)
    BIC = bic(l=l, n=n, p=1)
    #ICOMP = -2*l
    loglikuni = luni(m=m, n=n)
    #xisb = (m-aver)/(m-1)
    #llsb = cub.loglik(m=m, pi=1, xi=xisb, f=f)
    #nonzero = np.nonzero(f)[0]
    logliksat = lsat(n=n, f=f)
    # mean loglikelihood
    muloglik = l/n
    dev = 2*(logliksat-l)
    #LRT = 2*(l-llunif)
    theoric = pmf(m=m, sh=sh, delta=deltaest)
    #pearson = (f-n*theorpr)/np.sqrt(n*theorpr)
    #X2 = np.sum(pearson**2)
    #relares = (f/n-theorpr)/theorpr
    diss = dissimilarity(theoric,f/n)
    #FF2 = 1-diss
    #LL2 = 1/(1+np.mean((f/(n*theorpr)-1)**2))
    #II2 = (l-llunif)/(logsat-llunif)
    est_names = np.array(["delta"])
    e_types = np.array(["Shelter effect"])

    return CUBresCUSH(
    model="CUSH",
    m=m, n=n, sh=sh, theoric=theoric,
    est_names=est_names, e_types=e_types,
    estimates=np.array([deltaest]),
    stderrs=np.array([esdelta]),
    wald=np.array([wald]),
    pval=np.array([pval]),
    loglike=l, logliksat=logliksat,
    loglikuni=loglikuni, muloglik=muloglik,
    dev=dev, AIC=AIC, BIC=BIC,
    #ICOMP=ICOMP,
    seconds=(end-start).total_seconds(),
    time_exe=start,
    sample=sample, f=f, varmat=varmat,
    diss=diss,
    ass_pars=ass_pars,
    df=df, formula=formula
    )

class CUBresCUSH(CUBres):
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """
    def plot_ordinal(self, figsize=(7, 7), kind="bar",
        ax=None, saveas=None):
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
        
        R = choices(self.m)
        #print(R, self.f, self.n)
        delta = self.estimates[0]
        title = f"{self.model} model ($c={self.sh}$)   "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\delta={delta:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        if self.ass_pars is not None:
            delta_gen = self.ass_pars["delta"]
            title += "\n"
            title += fr"Gener($\delta={delta_gen:.3f}$)"
            p_gen = pmf(m=self.m, sh=self.sh, delta=delta_gen)
            ax.stem(R, p_gen, linefmt="--r",
                markerfmt="none", label="generator")
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
        ax.plot(R, self.theoric, ".--b",
            label="estimated", ms=10)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        ax.set_ylim((0, ax.get_ylim()[1]))
        
        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
    
    def plot_estim(self, ci=.95, ax=None,
        magnified=False, figsize=(7, 7), saveas=None):
        r"""Plots the estimated parameter values in the parameter space and
        the asymptotic standard error.
        
        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipse
        :type ci: float
        :param magnified: if False the limits will be the entire parameter space, otherwise let matplotlib choose the limits
        :type magnified: bool
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
        
        delta = self.estimates[0]
        ax.set_xlabel(fr"$\delta$  shelter effect (c={self.sh})")
        ax.plot(delta, 0,
            ".b",ms=20, alpha=.5,
            label="estimated")
        if self.ass_pars is not None:
            delta_gen = self.ass_pars["delta"]
            ax.scatter(delta_gen, 0,
                facecolor="None",
                edgecolor="r", s=200, label="generator")
        ax.set_yticks([])
        if not magnified:
            ax.set_xlim((0,1))
            ticks = np.arange(0, 1.1, .1)
            ax.set_xticks(ticks)
        if ci is not None:
            alpha = 1-ci
            z = abs(sps.norm().ppf(alpha/2))
            ax.plot(
                [delta-z*self.stderrs, delta+z*self.stderrs],
                [0, 0],
                "b", lw=1,
                label=f"CI {ci:.0%}"
            )
        ax.grid(True)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        
        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax

    def plot(self, ci=.95, saveas=None, figsize=(7, 15)):
        r"""Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the standard error
        :type ci: float
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        self.plot_ordinal(ax=ax[0])
        self.plot_estim(ax=ax[1], ci=ci)
        self.plot_estim(ax=ax[2], ci=ci,
            magnified=True)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
        
    # DEPRECATED
    def _old_plot(self,
        ci=.95,
        saveas=None,
        figsize=(7, 15)
        ):
        """
        :DEPRECATED:
        plot CUB model fitted from a sample
        """
        R = choices(self.m)
        #print(R, self.f, self.n)
        delta = self.estimates[0]
        title = f"{self.model} model    "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\delta={delta:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        #X2 = None

        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax[0].set_xticks(R)
        ax[0].set_xlabel("Ordinal")
        ax[0].set_ylabel("probability distribution")
        ax[1].set_xlim((0,1))
        #ax[1].set_ylim((0,1))
        ticks = np.arange(0, 1.1, .1)
        ax[1].set_xticks(ticks)
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[1].set_xlabel(r"$\delta$  shelter effect")
        ax[2].set_xlabel(r"$\delta$  shelter effect")

        # change all spines
        for axis in ['bottom']:
            for i in [1,2]:
                ax[i].spines[axis].set_linewidth(2)
                # increase tick width
                ax[i].tick_params(width=2)

        #p = pmf(m=self.m, sh=self.sh, delta=delta)
        ax[0].plot(R, self.theoric, ".--b",
            label="estimated", ms=10)
        ax[1].plot(delta, 0,
            ".b",ms=20, alpha=.5,
            label="estimated")
        ax[2].plot(delta, 0, 
            ".b",ms=20, alpha=.5,
            label="estimated")
        #ax[0].stem(R, p, linefmt="--r",
#            markerfmt="none", label="estimated")
#        ax[1].scatter(1-self.pi, 1-self.xi,
#            facecolor="None",
#            edgecolor="r", s=200, label="estimated")
#        ax[2].scatter(1-self.pi, 1-self.xi,
#            facecolor="None",
#            edgecolor="r", s=200, label="estimated")
        #if self.sample is not None:
        ax[0].scatter(R, self.f/self.n, 
            facecolor="None",
            edgecolor="k", s=200, label="observed")
        if self.ass_pars is not None:
            delta_gen = self.ass_pars["delta"]
            p_gen = pmf(m=self.m, sh=self.sh, delta=delta_gen)
            ax[0].stem(R, p_gen, linefmt="--r",
            markerfmt="none", label="generator")
            ax[1].scatter(delta_gen, 0,
            facecolor="None",
            edgecolor="r", s=200, label="generator")
            ax[2].scatter(delta_gen, 0,
            facecolor="None",
            edgecolor="r", s=200, label="generator")

            #X2 = chisquared(
            #    self.f,
            #    self.n*p_gen
            #)
            title += fr"    theoric($\delta={delta_gen:.3f}$)"
            #title += fr"    $\chi^2={X2:.1f}$"
            #if pi is not None and xi is not None:
            #diss = dissimilarity(p, self.f/self.n)
            ax[1].set_title(
                    f"dissimilarity = {self.diss:.4f}"
                )
        if ci is not None:
            alpha = 1-ci
            z = abs(sps.norm().ppf(alpha/2))
            for u in [1,2]:
                ax[u].plot(
                    [delta-z*self.stderrs, delta+z*self.stderrs],
                    [0, 0],
                    "b", lw=1
                )

        ax[0].set_ylim((0, ax[0].get_ylim()[1]))
        ax[0].set_title(title)
        ax[0].legend(loc="upper left",
        bbox_to_anchor=(1,1))
        ax[1].legend(loc="upper left",
        bbox_to_anchor=(1,1))
        ax[2].legend(loc="upper left",
        bbox_to_anchor=(1,1))
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig
