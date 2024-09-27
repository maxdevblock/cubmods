# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cub0w-module:

CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial) with covariates for the feeling component.

Description:
============
    This module contains methods and classes
    for CUB_0W model family.

Manual, Examples and References:
================================
    - `Models manual <manual.html#cub-with-covariates>`__
  
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
    r"""Average probability distribution of a specified CUB model 
    with covariates for the feeling component.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the vector of the probability distribution.
    :rtype: numpy array
    """
    n = W.shape[0]
    p = pmfi(m, pi, gamma, W)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, pi, gamma, W):
    r"""Probability distribution for each subject of a specified CUB model 
    with covariates for the feeling component.
    
    Auxiliary function of ``.draw()``.

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
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
    
    :math:`\Pr(R_i=r_i|\pmb\theta;\pmb T_i),\; i=1 \ldots n`
    
    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    p = pi*(bitgamma(sample=sample, m=m, W=W, gamma=gamma)-1/m) + 1/m
    return p

def _proba(m, pi, xi, r):
    """
    :DEPRECATED:
    """
    return None

def _cmf(m, pi, gamma, W): #TODO: test cmf
    r"""Average cumulative probability of a specified CUB model
    with covariates for the feeling component.

    :math:`\Pr(R_i \leq r | \pmb\theta ; \pmb w_i),\; r=1 \ldots m`
    
    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the array of the cumulative probability distribution.
    :rtype: numpy array
    """
    return pmf(m, pi, gamma, W).cumsum()

def _mean(m, pi, xi): #TODO mean
    return None

def _var(m, pi, xi): #TODO var
    return None

def _std(m, pi, xi): #TODO std
    return None

def _skew(pi, xi): #TODO skew
    return None

def _mean_diff(m, pi, xi): #TODO mean_diff
    return None
    
def _median(m, pi, xi): #TODO median
    return None
    
def _gini(m, pi, xi): #TODO gini
    return None
    
def _laakso(m, pi, xi): #TODO laakso
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
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the log-likelihood value
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
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the variance-covariance matrix of the CUB model
    :rtype: numpy ndarray
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
    r"""
    Preliminary parameter estimates of a CUB model with covariates for the feeling component.

    Compute preliminary parameter estimates for the feeling component of a CUB model 
    fitted to ordinal responses.
    These estimates are set as initial values for parameters to start the E-M algorithm.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: an array :math:`\pmb\gamma^{(0)}`
    :rtype: array of float
    """
    WW = np.c_[np.ones(W.shape[0]), W]
    ni = np.log((m-sample+.5)/(sample-.5))
    gamma = np.linalg.inv(WW.T @ WW) @ (WW.T @ ni)
    return gamma

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, pi, gamma, W,
    df, formula, seed=None):
    r"""
    Draw a random sample from a specified CUB model with covariates for
    the feeling component.

    :param m: number of ordinal categories
    :type m: int
    :param n: number of ordinal responses to be drawn
    :type n: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None;
        it must be :math:`\neq 0`
    :type seed: int, optional
    :return: an instance of ``CUBsample`` (see `here <cubmods.html#cubmods.smry.CUBsample>`__) containing ordinal responses drawn from the specified model
    """
    #np.random.seed(seed)
    assert len(gamma) == W.shape[1]+1
    n = W.shape[0]
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
    p_types = np.concatenate((
        ["Uncertainty"],
        np.repeat(["Feeling"], len(gamma))
    ))
    sample = CUBsample(
        model="CUB(0W)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        p_types=p_types,
        seed=seed, diss=diss,
        theoric=theoric, df=df,
        formula=formula
    )
    return sample

###################################################################
# INFERENCE
###################################################################
def effe01(gamma, esterno01, m):
    r"""
    Auxiliary function for the log-likelihood estimation of CUB models
    with covariates for the feeling component.

    Compute the opposite of the scalar function that is maximized when running 
    the E-M algorithm for CUB models with covariates for the feeling parameter.

    It is called as an argument for ``minimize`` within CUB function for models with covariates for
    feeling or for both feeling and uncertainty.

    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param esterno01: a matrix binding together: the vector :math:`\pmb\tau` of the posterior probabilities
        that each observation has been generated by the first component distribution of the mixture, 
        the ordinal data :math:`\pmb r` and the matrix :math:`\pmb w` of the selected covariates 
        accounting for an intercept term
    :param m: number of ordinal categories
    :type m: int
    :return: the expected value of the inconplete log-likelihood
    :rtype: float
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

def mle(sample, m, W, df, formula,
    ass_pars=None,
    maxiter=500,
    tol=1e-4):
    """
    Main function for CUB models with covariates for the feeling component.

    Function to estimate and validate a CUB model for given ordinal responses, with covariates for
    explaining the feeling component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
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
    :return: an instance of ``CUBresCUB0W`` (see the Class for details)
    :rtype: object
    """
    # validate parameters
    #if not validate_pars(m=m, n=sample.size):
    #    pass
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
            #W=W,
            diss=diss,
            df=df, formula=formula,
            ass_pars=ass_pars
            # pi_gen=pi_gen, xi_gen=xi_gen
        )
    return res

class CUBresCUB0W(CUBres):
    """Object returned by ``.mle()`` function.
    See `here <cubmods.html#cubmods.smry.CUBres>`__ the Base for details.
    """

    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar", #options bar, scatter
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
        title += f"Dissim(est,obs)={self.diss:.4f}"
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
            Wcols = self.est_names[
                -(1-len(self.ass_pars["gamma"])):
            ]
            ass_p = pmf(
                m=self.m,
                pi=self.ass_pars["pi"],
                gamma=self.ass_pars["gamma"],
                W=self.df[Wcols]
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
