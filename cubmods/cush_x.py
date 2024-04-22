"""
CUB models in Python.
Module for CUSH (Combination of Uniform
and Shelter effect) with covariates.

Description:
    This module contains methods and classes
    for CUSH model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    TODO: add example

References:
    * Capecchi, S., & Piccolo, D. (2017).
      Dealing with heterogeneity in ordinal responses.
      Quality & Quantity, 51, 2375-2393.
      DOI: 10.1007/s11135-016-0393-3

List of TODOs:
    TODO: check gini & laakso

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
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
    p = pmfi(m, sh, omega, X)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, sh, omega, X):
    delta = logis(X, omega)
    #print(delta)
    n = X.shape[0]
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = pmf_cush(m=m, sh=sh, 
            delta=delta[i])
    return p

def proba(m, sample, X, omega, sh):
    delta = logis(X, omega)
    D = (sample==sh).astype(int)
    p = delta*(D-1/m)+1/m
    return p

def draw(m, sh, omega, X, seed=None):
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
    
    return CUBsample(
        model="CUSH(X)",
        m=m, sh=sh,
        pars=omega,
        par_names=par_names,
        theoric=theoric,
        diss=diss,
        X=X,
        rv=rv.astype(int),
        seed=seed
    )

def loglik(m, sample, X, omega, sh):
    p = proba(m=m, sample=sample, X=X,
        omega=omega, sh=sh)
    l = np.sum(np.log(p))
    return l

def effe(pars, esterno, m, sh):
    sample = esterno[:,0]
    X = esterno[:,2:] # no 1
    l = loglik(m=m, sample=sample, X=X,
        omega=pars, sh=sh)
    return -l

def mle(m, sample, X, sh, gen_pars=None):
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
        X=X, diss=diss,
        seconds=(end-start).total_seconds(),
        time_exe=start, gen_pars=gen_pars
    )

class CUBresCUSHX(CUBres):
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        title = "MARGINAL PROBABILITY MASS\n"
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
