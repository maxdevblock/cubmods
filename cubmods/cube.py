# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cube000-module:

CUB models in Python.
Module for CUBE (Combination of Uniform
and Beta-Binomial).

Description:
============
    This module contains methods and classes
    for CUBE model family.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    :math:`\Pr(R=r|\pmb{\theta}) = \pi \beta e(\xi,\phi)+\dfrac{1-\pi}{m},\; r=1,2,\ldots,m`

Manual and Examples
==========================
    - `Models manual <manual.html#cube-without-covariates>`__

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
==============
  - TODO: adjust 3d plots legend

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
import scipy.stats as sps
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .general import (
    choices, freq, dissimilarity,
    conf_ell, luni, lsat, aic, bic,
    plot_ellipsoid
    #InvalidCategoriesError,
    #chisquared,
)
from . import cub
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def prob(m, pi, xi, phi, r):
    r"""Probability :math:`\Pr(R = r | \pmb\theta)` of a CUBE model without covariates.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param r: ordinal response
    :type r: int
    :return: the probability :math:`\Pr(R = r | \pmb\theta)` of a CUBE model without covariates.
    :rtype: numpy array
    """
    i = np.arange(0, m-1)
    # Pr(R=1)
    pBe = ((xi+i*phi)/(1+i*phi)).prod()
    for j in np.arange(1, r):
        pBe *= ((m-j)/j) * ((1-xi+(j-1)*phi)/(xi+(m-j-1)*phi))
    p = pi*pBe + (1-pi)/m
    return p

def betar(m, xi, phi):
    r"""Beta-Binomial distribution.

    Return the Beta-Binomial distribution with given parameters.

    :param m: number of ordinal categories
    :type m: int
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :return: array of length :math:`m` of the Beta-Binomial distribution.
    :rtype: numpy array
    """
    R = choices(m)
    km = np.arange(0, m-1)
    pBe = np.zeros(R.size)
    # Pr(R=1)
    pBe[0] = (1-(1-xi)/(1+phi*km)).prod()
    # Pr(R>1)
    for r in range(1,m):
        pBe[r] = pBe[r-1] * ((m-r)/r) * ((1-xi+(r-1)*phi)/(xi+(m-r-1)*phi))
    return pBe

def pmf(m, pi, xi, phi):
    r"""Probability distribution of a specified CUBE model.

    :math:`\Pr(R = r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :return: array of length :math:`m` of the distribution of a CUBE model without covariates.
    :rtype: numpy array
    """
    pBe = betar(m, xi, phi)
    ps = pi*(pBe-1/m) + 1/m
    return ps

def cmf(m, pi, xi, phi):
    r"""Cumulative probability of a specified CUBE model.

    :math:`\Pr(R \leq r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :return: array of length :math:`m` of the cumulative probability of a CUBE model without covariates.
    :rtype: numpy array
    """
    return pmf(m, pi, xi, phi).cumsum()

def mean(m, pi, xi):
    r"""Mean of a CUBE model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the expected value of the model
    :rtype: float
    """
    #_ = phi # CUBE mean does not depend on phi
    return (m+1)/2 + pi*(m-1)*(1/2-xi)

def var(m, pi, xi, phi):
    r"""Variance of a CUBE model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :return: the variance of the model
    :rtype: float
    """
    #v1 = pi*(m-1)*(m-2)*xi*(1-xi)*phi/(1+phi)
    #v2a = pi*xi*(1-xi)
    #v2b = (1-pi)*(m+1)/12
    #v2c = pi*(1-pi)*(m-1)*((1/2-xi)**2)
    #v = v1 + (m-1)*(v2a+v2b+v2c)
    v = cub.var(m,pi,xi) + pi*xi*(1-xi)*(m-1)*(m-2)*phi/(1+phi)
    return v

# TODO: check skew
def _skew(pi, xi, phi):
    r"""
    skewness normalized eta index
    """
    _ = phi #TODO: use phi or not?
    return pi*(1/2 - xi)

# TODO: test mean_diff
def _mean_diff(m, pi, xi, phi):
    R = choices(m)
    S = choices(m)
    mu = 0
    for r in R:
        for s in S:
            mu += abs(r-s)*prob(m,pi,xi,phi,r)*prob(m,pi,xi,phi,s)
    return mu
    
# TODO: test meadian
def _median(m, pi, xi, phi):
    R = choices(m)
    cp = cmf(m, pi, xi, phi)
    M = R[cp>.5][0]
    if M > R.max():
        M = R.max()
    return M

# TODO: test gini
def _gini(m, pi, xi, phi):
    ssum = 0
    for r in choices(m):
        ssum += prob(m, pi, xi, phi, r)**2
    return m*(1-ssum)/(m-1)

# TODO: test laakso
def _laakso(m, pi, xi, phi):
    g = _gini(m, pi, xi, phi)
    return g/(m - (m-1)*g)

def loglik(m, pi, xi, phi, f):
    r"""Log-likelihood function of a CUBE model without covariates.

    Compute the log-likelihood function of a CUBE model without covariates fitting 
    the given absolute frequency distribution.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param f: array of absolute frequency distribution
    :type f: array of int
    :return: the log-likelihood value
    :rtype: float
    """
    L = pmf(m, pi, xi, phi)
    l = (f*np.log(L)).sum()
    return l

def varcov(m, pi, xi, phi, sample):
    r"""Variance-covariance matrix for CUBE models based on the observed information matrix.

    Compute the variance-covariance matrix of parameter estimates for a CUBE model without covariates 
    as the inverse of the observed information matrix.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param sample: array of ordinal responses
    :type sample: array of int
    :return: the variance-covariance matrix of the CUBE model
    :rtype: numpy ndarray
    """
    R = choices(m)
    f = freq(sample, m)
    ### sum1; sum2; sum3; sum4; sum5; as in Iannario (2013), "Comm. in Stat. Theory & Methods"
    sum1=np.full(m, np.nan)
    sum2=np.full(m, np.nan)
    sum3=np.full(m, np.nan)
    sum4=np.full(m, np.nan)
    d1=np.full(m, np.nan)
    d2=np.full(m, np.nan)
    h1=np.full(m, np.nan)
    h2=np.full(m, np.nan)
    h3=np.full(m, np.nan)
    h4=np.full(m, np.nan)
    #np.zeros(m)
    # Pr(R=r|pi,xi)
    pr = pmf(m, pi, xi, phi)
    for jr in R:
        arr1 = np.arange(jr)
        arr2 = np.arange(m-jr+1)
        seq1 = 1/((1-xi)+phi*arr1)
        seq2 = 1/((xi)+phi*arr2)
        #print("########### jr", jr)
        #print("arr", arr1, arr2)
        #print("seq", seq1, seq2)
        seq3 = arr1/((1-xi)+phi*arr1)
        seq4 = arr2/((xi)+phi*arr2)
        dseq1 = seq1**2
        dseq2 = seq2**2
        hseq1 = dseq1*arr1
        hseq2 = dseq2*arr2
        hseq3 = dseq1*arr1**2
        hseq4 = dseq2*arr2**2
        #############
        sum1[jr-1] = np.sum(seq1)-seq1[jr-1]
        sum2[jr-1] = np.sum(seq2)-seq2[m-jr]
        #print("sum", sum1, sum2)
        sum3[jr-1] = np.sum(seq3)-seq3[jr-1]
        sum4[jr-1] = np.sum(seq4)-seq4[m-jr]
        d1[jr-1] = np.sum(dseq1)-dseq1[jr-1]
        d2[jr-1] = -(np.sum(dseq2)-dseq2[m-jr])
        h1[jr-1] = -(np.sum(hseq1)-hseq1[jr-1])
        h2[jr-1] = -(np.sum(hseq2)-hseq2[m-jr])
        h3[jr-1] = -(np.sum(hseq3)-hseq3[jr-1])
        h4[jr-1] = -(np.sum(hseq4)-hseq4[m-jr])

    arr3 = np.arange(0, m-1) #(0:m-2)
    seq5 = arr3/(1+phi*arr3)
    sum5 = np.sum(seq5)
    h5 = -np.sum(seq5**2)
    ### Symbols as in Iannario (2013), "Comm. in Stat.", ibidem (DP notes)
    uuur = 1-1/(m*pr)
    ubar = uuur+pi*(1-uuur)
    vbar = ubar-1
    aaar = sum2-sum1
    #print("sums aaar", sum2, sum1, aaar)
    bbbr = sum3+sum4-sum5
    cccr = h3+h4-h5
    dddr = h2-h1
    eeer = d2-d1
    ###### dummy product
    prodo = f*ubar
    ######
    infpipi = np.sum(f*uuur**2)/pi**2
    infpixi = np.sum(prodo*(uuur-1)*aaar)/pi
    infpiphi = np.sum(prodo*(uuur-1)*bbbr)/pi
    infxixi = np.sum(prodo*(vbar*aaar**2-eeer))
    infxiphi = np.sum(prodo*(vbar*aaar*bbbr-dddr))
    infphiphi = np.sum(prodo*(vbar*bbbr**2-cccr))
    ### Information matrix
    infmat = np.zeros(shape=(3,3))
    infmat[0,0] = infpipi
    infmat[0,1] = infpixi
    infmat[0,2] = infpiphi
    infmat[1,0] = infpixi
    infmat[1,1] = infxixi
    infmat[1,2] = infxiphi
    infmat[2,0] = infpiphi
    infmat[2,1] = infxiphi
    infmat[2,2] = infphiphi

    varmat = np.ndarray(shape=(3,3))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    return varmat

# TODO: .5 o .3?
def init_theta(sample, m):
    r"""Naive estimates for CUBE models without covariates.

    Compute naive parameter estimates of a CUBE model without covariates for given ordinal responses. 
    These preliminary estimators are used within the package code to start the E-M algorithm.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :type m: int
    :return: a tuple of :math:`(\pi^{(0)}, \xi^{(0)}, \phi^{(0)})`
    :rtype: tuple of float
    """
    f = freq(sample, m)
    pi, xi = cub.init_theta(f, m)
    varsam = np.mean(sample**2) - np.mean(sample)**2
    varcub = cub.var(m, pi, xi)
    phi = min(
        max(
            (varcub-varsam)/(-pi*xi*(1-xi)*(m-1)*(m-2)-varcub+varsam),
            .01
        ), .5 #qui...!
    )
    return pi, xi, phi

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, pi, xi, phi, n,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUBE model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
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
    np.random.seed(seed)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=pmf(m, pi, xi, phi)
        )
    pars = np.array([pi, xi, phi])
    par_names = np.array([
        'pi', 'xi', 'phi'
    ])
    f = freq(m=m, sample=rv)
    theoric = pmf(m=m, pi=pi, xi=xi, phi=phi)
    diss = dissimilarity(f/n, theoric)
    sample = CUBsample(
        model="CUBE",
        rv=rv, m=m,
        pars=pars,
        par_names=par_names,
        theoric=theoric,
        diss=diss, df=df,
        formula=formula
    )
    return sample

###################################################################
# INFERENCE
###################################################################

def effecube(params, tau, f, m):
    r"""Auxiliary function for the log-likelihood estimation of CUBE models without covariates.

    Define the opposite of the scalar function that is maximized when running the E-M 
    algorithm for CUBE models without covariates.

    :param params: array of initial estimates for the feeling and the overdispersion parameters
    :type params: array of float
    :param tau: a column vector of length :math:`m` containing the posterior
        probabilities that each observed category has been generated by the first component distribution 
        of the mixture
    :type tau: array
    :param f: array of the absolute frequencies of the observations
    :type f: array
    :param m: number of ordinal categories
    :type m: int
    :return: the expected value of the inconplete log-likelihood
    :rtype: float
    """
    xi = params[0]
    phi = params[1]
    pBe = betar(m, xi, phi)
    return -np.sum(tau*f*np.log(pBe))

def mle(sample, m, df, formula,
    ass_pars=None,
    maxiter=1000,
    tol=1e-6):
    r"""Main function for CUBE models without covariates.

    Estimate and validate a CUBE model without covariates.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
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
    :return: an instance of ``CUBresCUBE`` (see the Class for details)
    :rtype: object    
    """
    # validate parameters
    #if not validate_pars(m=m, n=sample.size):
    #    pass
    # tta = [] # test optimize time
    # ttb = [] # test optimize time
    # ttc = [] # test optimize time
    # ttd = [] # test optimize time
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

    # initialize (pi, xi)
    pi, xi, phi = init_theta(sample, m)
    # compute loglikelihood
    l = loglik(m, pi, xi, phi, f)

    # start E-M algorithm
    niter = 1
    while niter < maxiter:
        # tta.append(dt.datetime.now()) # test optimize time
        lold = l
        bb = betar(m, xi, phi)
        aa = (1-pi)/(m*pi*bb)
        tau = 1/(1+aa)
        pi = np.sum(f*tau)/n
        #params = (xi, phi)
        #TODO: upper lower maxiter?
        optim = minimize(
            effecube, x0=[xi, phi], args=(tau, f, m),
             method="L-BFGS-B",
             bounds=[(.01, .99), (.01, .3)],
             options={
                 "maxiter":100,
                 #"ftol":1.49e-8,
                 #"gtol":1.49e-8,
                 #"maxls":5,
                 #"maxfun":5,
                },
        )
        # ttc.append(dt.datetime.now()) # test optimize time
        xi = optim.x[0]
        phi = optim.x[1]
        #print(optim.x)
        # avoid division by zero
        if pi < .001:
            pi = .001
            niter = maxiter-1
        if pi > .999:
            pi = .99
        # new lohlikelihood
        lnew = loglik(m, pi, xi, phi, f)
        # compute delta-loglik
        deltal = abs(lnew-lold)
        # ttd.append(dt.datetime.now()) # test optimize time
        # check tolerance
        if deltal <= tol:
            break
        else:
            l = lnew
        niter += 1
    # end E-M algorithm

    # tta = np.array(tta) # test optimize time
    # ttb = np.array(ttb) # test optimize time
    # ttc = np.array(ttc) # test optimize time
    # ttd = np.array(ttd) # test optimize time
    # precalc = (ttb-tta).sum().total_seconds() # test optimize time
    # optimiz = (ttc-ttb).sum().total_seconds() # test optimize time
    # postcal = (ttd-ttc).sum().total_seconds() # test optimize time

    l = lnew
    # variance-covariance matrix
    #print("est", pi, xi, phi)
    varmat = varcov(m=m, pi=pi, xi=xi, phi=phi, sample=sample)
    end = dt.datetime.now()
    # standard errors
    stderrs = np.array([
        np.sqrt(varmat[0,0]),
        np.sqrt(varmat[1,1]),
        np.sqrt(varmat[2,2])
    ])
    # Wald statistics
    wald = np.array([pi, xi, phi])/stderrs
    # p-value
    pval = 2*(sps.norm().sf(abs(wald)))
    # Akaike Information Criterion
    AIC = aic(l=l, p=3)
    # Bayesian Information Criterion
    BIC = bic(l=l, p=3, n=n)
    # mean loglikelihood
    muloglik = l/n
    # loglik of null model (uniform)
    loglikuni = luni(m=m, n=n)
    # loglik of saturated model
    logliksat = lsat(f=f, n=n)
    # # loglik of shiftet binomial
    # xibin = (m-sample.mean())/(m-1)
    # loglikbin = loglik(m, 1, xibin, f)
    # # Explicative powers
    # Ebin = (loglikbin-loglikuni)/(logliksat-loglikuni)
    # Ecub = (l-loglikbin)/(logliksat-loglikuni)
    # Ecub0 = (l-loglikuni)/(logliksat-loglikuni)
    # deviance from saturated model
    dev = 2*(logliksat-l)
    # ICOMP metrics
    #npars = 3
    #trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + npars*np.log(trvarmat/npars) - np.log(np.linalg.det(varmat))
    theoric = pmf(m=m, pi=pi, xi=xi, phi=phi)
    diss = dissimilarity(f/n, theoric)
    estimates = np.concatenate((
        [pi], [xi], [phi]
    ))
    est_names = np.array(["pi", "xi", "phi"])
    e_types = np.array([
        "Uncertainty", "Feeling",
        "Overdispersion"
    ])

    # results object
    res = CUBresCUBE(
            model="CUBE",
            m=m, n=n, niter=niter,
            maxiter=maxiter, tol=tol,
            theoric=theoric,
            est_names=est_names,
            e_types=e_types,
            estimates=estimates,
            stderrs=stderrs,
            pval=pval, wald=wald,
            loglike=l, muloglik=muloglik,
            loglikuni=loglikuni,
            logliksat=logliksat,
            #loglikbin=loglikbin,
            #Ebin=Ebin, Ecub=Ecub, Ecub0=Ecub0,
            dev=dev, AIC=AIC, BIC=BIC,
            #ICOMP=ICOMP,
            seconds=(end-start).total_seconds(),
            # precalc=precalc, # test optimize time
            # optimiz=optimiz, # test optimize time
            # postcal=postcal, # test optimize time
            time_exe=start,
            #rho=rho,
            sample=sample, f=f,
            varmat=varmat,
            diss=diss,
            ass_pars=ass_pars,
            df=df, formula=formula
        )
    return res

class CUBresCUBE(CUBres):
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """

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
        
        pi = self.estimates[0]
        xi = self.estimates[1]
        phi = self.estimates[2]
        title = f"{self.model} model    "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$ , $\phi={phi:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.3f}"
        #TODO: add dissimilarity from generating model
        if self.ass_pars is not None:
            title += "\n"
            title += fr"Gener($\pi={self.ass_pars['pi']:.3f}$ , $\xi={self.ass_pars['xi']:.3f}$ , "
            title += fr"$\phi={self.ass_pars['phi']:.3f}$)"
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
        if self.ass_pars is not None:
            pi_gen = self.ass_pars["pi"]
            xi_gen = self.ass_pars["xi"]
            phi_gen = self.ass_pars["phi"]
            p_gen = pmf(m=self.m, pi=pi_gen, xi=xi_gen, phi=phi_gen)
            ax.stem(R, p_gen, linefmt="--r",
            markerfmt="none", label="generating")

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax

    #TODO: add option to show displacement from CUB model?
    def _plot_confell(self,
        figsize=(7, 5),
        ci=.95,
        equal=True,
        magnified=False,
        confell=False,
        ax=None,
        saveas=None
        ):
        """
        :DEPRECATED:
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        else:
            fig = None
        
        pi = self.estimates[0]
        xi = self.estimates[1]
        phi = self.estimates[2]

        if equal:
            ax.set_aspect("equal")
        if self.rho is not None:
            ax.set_title(f"Corr(pi,xi)= {self.rho}")

        ax.set_xlabel(r"$(1-\pi)$  uncertainty")
        ax.set_ylabel(r"$(1-\xi)$  feeling")

        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)

        ax.plot(1-pi, 1-xi,
            ".b", ms=20, alpha=.5,
            label="estimated")
        ax.text(1-pi, 1-xi,
            fr"  $\phi = {phi:.3f}$" "\n",
            ha="left", va="bottom")
        if self.ass_pars is not None:
            pi_gen = self.ass_pars["pi"]
            xi_gen = self.ass_pars["xi"]
            #phi_gen = self.ass_pars["phi"]
            ax.scatter(1-pi_gen, 1-xi_gen,
                facecolor="None",
                edgecolor="r", s=200, label="generating")
        if confell:
            # Confidence Ellipse
            conf_ell(
                self.varmat,
                1-pi, 1-xi,
                ci, ax
            )

        if not magnified:
            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            ticks = np.arange(0, 1.05, .1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if xlim[0] < 0:
                ax.set_xlim((0, xlim[1]))
                xlim = ax.get_xlim()
            if xlim[1] > 1:
                ax.set_xlim((xlim[0], 1))
            if ylim[0] < 0:
                ax.set_ylim((0, ylim[1]))
                ylim = ax.get_ylim()
            if ylim[1] > 1:
                ax.set_ylim((ylim[0], 1))
            # if self.rho is not None:
            #     beta1 = self.varmat[0,1] / self.varmat[0,0]
            #     ax.axline(
            #         [1-self.pi, 1-self.xi],
            #         slope=beta1, ls="--"
            #     )

        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        ax.grid(visible=True)

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas,
                    bbox_inches='tight')
        return fig, ax

    def plot3d(self, ax, ci=.95,
        magnified=False):
        r"""Plots the estimated parameter values in the parameter space and
        the asymptotic confidence ellipsoid with its projections.
        
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipsoid
        :type ci: float
        :param magnified: if False the limits will be the entire parameter space, otherwise let matplotlib choose the limits
        :type magnified: bool
        :param ax: matplotlib axis, if None a new figure will be created, defaults to None
        :type ax: matplolib ax, optional
        """
        pi = self.estimates[0]
        xi = self.estimates[1]
        ph = self.estimates[2]
        V = self.varmat
        #print()
        #print("VARCOV(pxf)")
        #print(V)
        #espxf = np.sqrt(
        #            np.diag(V))
        #print()
        #print("ES(pxf)")
        #print(espxf)
        plot_ellipsoid(V=V,
            E=(1-pi,1-xi,ph), ax=ax,
            zlabel=r"Overdispersion $\phi$",
            magnified=magnified, ci=ci
        )


    def plot(self,
        ci=.95,
        saveas=None,
        confell=False,
        test3=True,
        figsize=(7, 15)
        ):
        r"""Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipsoid
        :type ci: float
        :param confell: **DEPRECATED**, defaults to False
        :type confell: bool
        :param test3: **DEPRECATED**, defaults to True
        :type test3: bool
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        fig, ax = plt.subplots(3, 1,
            figsize=figsize,
            constrained_layout=True)
        self.plot_ordinal(ax=ax[0])
        if test3:
            ax[1].remove()
            ax[2].remove()
            ax[1] = fig.add_subplot(3,1,2,
                projection='3d')
            ax[2] = fig.add_subplot(3,1,3,
                projection='3d')
            self.plot3d(ax=ax[1], ci=ci)
            self.plot3d(ax=ax[2], ci=ci,
                magnified=True)
        else:
            self._plot_confell(ci=ci,
            ax=ax[1], confell=confell)
            self._plot_confell(
                ci=ci, ax=ax[2],
                confell=confell,
                magnified=True, equal=False)
            plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
