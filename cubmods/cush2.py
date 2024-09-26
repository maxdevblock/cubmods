# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cush200-module:

CUB models in Python.
Module for CUSH2 (Combination of Uniform
and 2 Shelter Choices).

Description:
============
    This module contains methods and classes
    for CUSH2 model family.

Manual, Examples and References:
================================
    - `Models manual <manual.html#cush2-without-covariates>`__
  
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
#from scipy.optimize import minimize
#import seaborn as sns
import matplotlib.pyplot as plt
from .general import (
    conf_ell, freq, dissimilarity,
    choices, aic, bic, luni, lsat,
    #NoShelterError
)
from .smry import CUBres, CUBsample

def pmf(m, c1, c2, d1, d2):
    r"""Probability distribution of a specified CUSH2 model.

    :math:`\Pr(R = r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param c1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type c1: int
    :param c2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type c2: int
    :param d1: 1st shelter choice parameter :math:`\delta_1`
    :type d1: float
    :param d2: 2nd shelter choice parameter :math:`\delta_2`
    :type d2: float
    :return: the probability distribution
    :rtype: array
    """
    R = choices(m)
    D1 = (R==c1).astype(int)
    D2 = (R==c2).astype(int)
    p = d1*D1 + d2*D2 + (1-d1-d2)/m
    #p = np.zeros(m)
    #for i in R:
    #    if i == c1:
    #        p[i-1] = d1 + (1-d1-d2)/m
    #    elif i == c2:
    #        p[i-1] = d2 + (1-d1-d2)/m
    #    else:
    #        p[i-1] = (1-d1-d2)/m
    return p

def draw(m, sh1, sh2, df, formula,
    delta1, delta2, n, seed=None):
    r"""Draw a random sample from a specified CUSH2 model.

    :param m: number of ordinal categories
    :type m: int
    :param sh1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type sh1: int
    :param sh2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type sh2: int
    :param delta1: 1st shelter choice parameter :math:`\delta_1`
    :type delta1: float
    :param delta2: 2nd shelter choice parameter :math:`\delta_2`
    :type delta2: float
    :param n: number of ordinal responses
    :type n: int
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` (see `here <cubmods.html#cubmods.smry.CUBsample>`__) containing ordinal responses drawn from the specified model
    """
    #if sh is None:
    #    raise NoShelterError(model="cush2")
    #c1 = sh[0]; c2 = sh[1]
    if delta1+delta2 > 1:
        raise Exception("delta1+delta2>1")
    theoric = pmf(m, sh1, sh2, delta1, delta2)
    np.random.seed(seed)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=theoric
        )
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    pars = np.array([delta1, delta2])
    par_names = np.array(["delta1", "delta2"])
    p_types = np.array(["Shelter1", "Shelter2"])
    #sh=np.array([c1, c2])
    sample = CUBsample(
        model="CUSH2",
        rv=rv, m=m,
        sh=np.array([sh1, sh2]),
        pars=pars,
        par_names=par_names,
        p_types=p_types,
        seed=seed, theoric=theoric,
        diss=diss, df=df,
        formula=formula
    )
    return sample

def varcov(m, n, d1, d2, fc1, fc2):
    r"""Compute the variance-covariance matrix of parameter 
    estimates of a CUSH2 model without covariates.

    :param m: number of ordinal categories
    :type m: int
    :param n: number of ordinal responses
    :type n: int
    :param d1: 1st shelter choice parameter :math:`\delta_1`
    :type d1: float
    :param d2: 2nd shelter choice parameter :math:`\delta_2`
    :type d2: float
    :param fc1: relative frequency of 1st shelter choice
    :type fc1: float
    :param fc2: relative frequency of 2nd shelter choice
    :type fc2: float
    :return: the variance-covariance matrix
    :rtype: numpy ndarray
    """
    I11 = n*(
        fc2*(m-1)**2 / (1-d1+d2*(m-1))**2 +
        fc1          / (1-d2+d1*(m-1))**2 +
        (1-fc1-fc2)  / (1-d1-d2)**2
        )
    I22 = n*(
        fc1*(m-1)**2 / (1-d2+d1*(m-1))**2 +
        fc2          / (1-d1+d2*(m-1))**2 +
        (1-fc1-fc2)  / (1-d1-d2)**2
    )
    I12 = n*(
        fc1*(m-1)    / (1-d2+d1*(m-1))**2 +
        fc2*(m-1)    / (1-d1+d2*(m-1))**2 -
        (1-fc1-fc2)  / (1-d1-d2)**2
    )
    infmat = np.array([
        [I11, I12],
        [I12, I22]
        ])
    varmat = np.ndarray(shape=(2,2))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    return varmat

def mle(sample, m, c1, c2,
    df, formula, ass_pars=None,
    maxiter=None, tol=None #for GEM compatibility
    ):
    r"""Main function for CUSH2 models without covariates.

    Estimate and validate a CUSH2 model for ordinal responses, without covariates.
    
    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param c1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type c1: int
    :param c2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type c2: int
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
    :return: an instance of ``CUBresCUSH2`` (see the Class for details)
    :rtype: object
    """
    _, _ = maxiter, tol
    start = dt.datetime.now()
    n = sample.size
    f = freq(sample=sample, m=m)
    fc1 = (sample==c1).sum() / n
    fc2 = (sample==c2).sum() / n
    d1 = (fc1*(m-1)+fc2-1)/(m-2)
    d1 = max([.01, d1])
    d2 = (fc2*(m-1)+fc1-1)/(m-2)
    d2 = max([.01, d2])
    varmat = varcov(m, n, d1, d2, fc1, fc2)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.array([d1, d2])
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    est_names = np.array(["delta1", "delta2"])
    e_types = np.array([
        "Shelter effects", None
    ])
    loglikuni = luni(m=m, n=n)
    logliksat = lsat(n=n, f=f)
    l = loglik(sample=sample, m=m,
        c1=c1, c2=c2)
    AIC = aic(l=l, p=2)
    BIC = bic(l=l, p=2, n=n)
    theoric = pmf(m, c1, c2, d1, d2)
    diss = dissimilarity(f/n, theoric)
    muloglik = l/n
    dev = 2*(logliksat-l)
    end = dt.datetime.now()
    
    return CUBresCUSH2(
        model="CUSH2",
        m=m, n=n, sh=np.array([c1, c2]),
        estimates=estimates,
        est_names=est_names,
        e_types=e_types,
        stderrs=stderrs, pval=pval,
        theoric=theoric,
        wald=wald, loglike=l,
        muloglik=muloglik,
        loglikuni=loglikuni,
        logliksat=logliksat,
        dev=dev, AIC=AIC, BIC=BIC,
        sample=sample, f=f, varmat=varmat,
        diss=diss,
        seconds=(end-start).total_seconds(),
        time_exe=start,
        ass_pars=ass_pars,
        df=df, formula=formula
    )

def loglik(sample, m, c1, c2):
    r"""Log-likelihood function for a CUSH2 model without covariates.

    Compute the log-likelihood function for a CUSH2 model 
    without covariate for the given ordinal responses.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param c1: Category corresponding to the 1st shelter choice :math:`[1,m]`
    :type c1: int
    :param c2: Category corresponding to the 2nd shelter choice :math:`[1,m]`
    :type c2: int
    :return: the log-likehood value
    :rtype: float
    """
    #l = (f*np.log(pr(m, d1, d2))).sum()
    n = sample.size
    fc1 = (sample==c1).sum()/n
    fc2 = (sample==c2).sum()/n
    fc3 = 1-fc1-fc2
    l = n*(fc1*np.log(fc1) + 
        fc2*np.log(fc2) + 
        fc3*np.log(fc3/(m-2)))
    return l

def _effe(d, sample, m, c1, c2):
    r"""
    """
    n = sample.size
    d1 = d[0]
    d2 = d[1]
    d3 = 1 - d1 - d2
    fc1 = (sample==c1).sum()/n
    fc2 = (sample==c2).sum()/n
    fc3 = 1 - fc1 - fc2
    l = n*(
        fc1*np.log(d1+d3/m) +
        fc2*np.log(d2+d3/m) +
        fc3*np.log(d3/m)
        )
    return -l

class CUBresCUSH2(CUBres):
    """Object returned by ``.mle()`` function.
    See `here <cubmods.html#cubmods.smry.CUBres>`__ the Base for details.
    """
    def plot_par_space(self,
        figsize=(7, 5),
        ax=None, ci=.95,
        saveas=None):
        r"""Plots the estimated parameter values in the parameter space and
        the asymptotic standard error.
        
        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipse
        :type ci: float
        :param ax: matplotlib axis, if None a new figure will be created, defaults to None
        :type ax: matplolib ax, optional
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        estd1, estd2 = self.estimates
        c1, c2 = self.sh

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        else:
            fig = None

        if self.ass_pars is not None:
            d1 = self.ass_pars["delta1"]
            d2 = self.ass_pars["delta2"]
            ax.plot(d1, d2, "xr",
                label="assumed",
                zorder=np.inf)

        ax.plot(estd1, estd2, "o", label="estimated")
        #ax.axhline(1-estd1-estd2, color="C1", ls="--", zorder=-1,
        #            label=r"$1-\hat\delta_1-\hat\delta_2$")
        #ax.axvline(1-estd1-estd2, color="C1", ls="--", zorder=-1)
        ax.fill_between([0,1], [1,0], [1,1], color="w", zorder=2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axline([0,1], slope=-1, color="k", lw=.75)
        conf_ell(self.varmat, estd1, estd2, ci, ax)
        ax.set_xlabel(fr"$\hat\delta_1$   for $c_1={c1}$")
        ax.set_xlim(0,1)
        ax.set_ylabel(fr"$\hat\delta_2$   for $c_2={c2}$")
        ax.set_ylim(0,1)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0,10.1,1)/10)
        ax.set_yticks(np.arange(0,10.1,1)/10)
        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_title("CUSH2 model parameter space")

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
    
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
        
        estd1, estd2 = self.estimates
        title = f"{self.model} model "
        title += fr"($c_1={self.sh[0]}$ , $c_2={self.sh[1]}$)"
        title += f"    $n={self.n}$\n"
        title += fr"Estim($\delta_1={estd1:.3f}$ , $\delta_2={estd2:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        if self.ass_pars is not None:
            title += "\n"
            title += fr"Assumed($\delta_1={self.ass_pars['delta1']:.3f}$ , $\delta_2={self.ass_pars['delta2']:.3f}$)"
            p_gen = pmf(c1=self.sh[0], c2=self.sh[1], d1=estd1, d2=estd2, m=self.m)
            R = choices(m=self.m)
            ax.stem(R, p_gen, linefmt="--r",
                markerfmt="none", label="assumed")
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
        ci=.95,
        saveas=None,
        figsize=(7, 11)
        ):
        r"""Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the standard error
        :type ci: float
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        fig, ax = plt.subplots(2, 1,
            figsize=figsize)
        self.plot_ordinal(ax=ax[0])
        self.plot_par_space(ax=ax[1],
            ci=ci)
        plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
