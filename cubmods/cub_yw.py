# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements
"""
CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial) with covariates.

Description:
============
    This module contains methods and classes
    for CUB_YW model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Reference Guide and Manual
==========================
  - Manual https://github.com/maxdevblock/cubmods/blob/main/Manual/02_cub_family.md
  - Reference Guide https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md


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

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
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
    """
    PMF of CUB model.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#pmfm-beta-gamma-y-w
    """
    p = pmfi(m, beta, gamma, Y, W)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, beta, gamma, Y, W):
    """
    Probability mass for each subject given parameters and covariates.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#pmfim-beta-gamma-y-w
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
    """
    Probability distribution of a CUB model with covariates for both feeling and uncertainty

    Compute the probability distribution of a CUB model with covariates for both the feeling 
    and the uncertainty components.
    """
    p = (
        logis(Y=Y, param=beta)*
        (bitgamma(sample=sample,m=m,
            W=W,gamma=gamma)-1/m)
        +1/m
    )
    return p

def draw(m, n, beta, gamma, Y, W, seed=None):
    """
    Draw a random sample from CUB model

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#drawm-n-beta-gamma-y-w
    """
    #np.random.seed(seed)
    assert n == W.shape[0]
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
    sample = CUBsample(
        model="CUB(YW)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, W=W, diss=diss,
        theoric=theoric
    )
    return sample

def loglik(m, sample, Y, W, beta, gamma):
    """
    Log-likelihood function of a CUB model with covariates for both feeling and uncertainty

    Compute the log-likelihood function of a CUB model fitting ordinal data
    with covariates for explaining both the feeling and the uncertainty components.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#loglikm-sample-y-w-beta-gamma
    """
    p = prob(m, sample, Y, W, beta, gamma)
    l = np.sum(np.log(p))
    return l

def varcov(m, sample, Y, W, beta, gamma):
    """
    Variance-covariance matrix of a CUB model with covariates for both uncertainty and feeling

    Compute the variance-covariance matrix of parameter estimates of a CUB model with covariates for
    both the uncertainty and the feeling components.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#varcovm-sample-y-w-beta-gamma
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
    gen_pars=None,
    maxiter=500,
    tol=1e-4):
    """
    Main function for CUB models with covariates for both the uncertainty and the feeling components

    Estimate and validate a CUB model for given ordinal responses, with covariates for explaining both the
    feeling and the uncertainty components by means of logistic transform.

    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#mlesample-m-y-w
    """
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
        varmat=varmat, Y=Y, W=W,
        diss=diss,
        gen_pars=gen_pars
        #dev=dev
    )

class CUBresCUBYW(CUBres):
    """
    https://github.com/maxdevblock/cubmods/blob/main/Manual/Reference%20Guide/cub_yw.md#cubrescubyw
    """
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
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
