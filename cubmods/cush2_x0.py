"""
CUB models in Python.
Module for 2-CUSH (Combination of Uniform
and 2 Shelter Choices) with covariates.

Description:
    This module contains methods and classes
    for 2-CUSH model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import cush2

    samp = pd.read_csv("ordinal.csv")
    fit = cush2.mle(samp.rv, m=7)
    print(fit.summary())
    fit.plot()
    plt.show()

References:
    * TODO: add references

List of TODOs:
    * 

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""

import datetime as dt
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_hess
from .general import (
    logis, colsof, aic, bic, lsat, luni,
    freq, dissimilarity, choices,
    lsatcov
)
from .cush2 import pmf as pmf_cush2
from .smry import CUBres, CUBsample

def pmfi(m, sh1, sh2,
    omega1, delta2,
    X1):
    delta1 = logis(X1, omega1)
    n = X1.shape[0]
    p_i = np.ndarray(shape=(n,m))
    for i in range(n):
        p_i[i] = pmf_cush2(m=m, c1=sh1,
            c2=sh2, d1=delta1[i],
            d2=delta2)
    return p_i

def pmf(m, sh1, sh2,
    omega1, delta2,
    X1):
    p_i = pmfi(m, sh1, sh2, omega1, delta2,
        X1)
    p = p_i.mean(axis=0)
    return p

def draw(m, sh1, sh2, omega1, delta2, X1,
    seed=None): #TODO: test draw
    n = X1.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m, sh1, sh2, omega1,
        delta2, X1)
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
        omega1, [delta2]
    ))
    par_names = np.concatenate((
        ["constant"], X1.columns,
        ["delta2"]
    ))
    return CUBsample(
        model="CUSH2(X1,0)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, X=X1, diss=diss,
        theoric=theoric, sh=[sh1, sh2]
    )

def loglik(sample, m, sh1, sh2,
    omega1, delta2,
    X1):
    delta1 = logis(X1, omega1)
    D1 = (sample==sh1).astype(int)
    D2 = (sample==sh2).astype(int)
    l = np.sum(np.log(
        delta1*D1 + delta2*D2 +
        (1-delta1-delta2)/m
    ))
    return l

def effe(pars, sample, m,
    sh1, sh2, X1):
    #w1 = colsof(X1)+1
    omega1 = pars[:-1]
    delta2 = pars[-1]
    l = loglik(sample, m, sh1, sh2,
        omega1, delta2,
        X1)
    return -l

#TODO: constraint (1-d1-d2)<1 ?
def mle(sample, m, sh1, sh2,
    X1, gen_pars=None,
    maxiter=None, tol=None):
    start = dt.datetime.now()
    w1 = colsof(X1)
    n = sample.size
    f = freq(m=m, sample=sample)
    fc1 = (sample==sh1).sum()/n
    fc2 = (sample==sh2).sum()/n
    delta1_0 = max([
        .01, (fc1*(m-1)+fc2-1)/(m-2)])
    om1_0 = np.log(delta1_0/(1-delta1_0))
    om1 = np.concatenate((
        [om1_0], np.repeat(.1, w1)
    ))
    delta2 = max([
        .01, (fc2*(m-1)+fc1-1)/(m-2)])

    pars = np.concatenate((om1, [delta2]))
    bounds = [[None,None] for _ in range(w1+1)]
    bounds.append([.01,.99])
    optim = minimize(
        effe, x0=pars,
        args=(sample, m,
            sh1, sh2, X1),
        bounds=bounds
    )
    estimates = optim.x
    omega1 = estimates[:-1]
    delta2 = estimates[-1]
    est_names = np.concatenate((
        ["constant"],
        [x for x in X1.columns],
        ["delta2"]
    ))
    e_types = np.concatenate((
        ["Shelter effects"],
        [None for _ in X1.columns],
        [None]
    ))
    
    infmat = approx_hess(estimates, effe,
        args=(sample, m, sh1,
            sh2, X1))
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
    #varmat = np.linalg.inv(apphess)
    stderrs = np.sqrt(np.diag(varmat))
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    theoric = pmf(m=m, sh1=sh1, sh2=sh2,
        omega1=omega1, delta2=delta2, X1=X1)
    diss = dissimilarity(f/n, theoric)
    l = loglik(m=m, sample=sample, sh1=sh1,
        sh2=sh2, omega1=omega1,
        delta2=delta2, X1=X1)
    AIC = aic(l=l, p=estimates.size)
    BIC = bic(l=l, p=estimates.size, n=n)
    #logliksat = lsat(n=n, f=f)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[X1]
    #)
    loglikuni = luni(m=m, n=n)
    #dev = 2*(logliksat-l)
    muloglik = l/n
    end = dt.datetime.now()
    
    return CUBresCUSH2X0(
        model="2CUSH(X0)",
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
        diss=diss,
        seconds=(end-start).total_seconds(),
        time_exe=start
    )

class CUBresCUSH2X0(CUBres):
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        title = f"{self.model} model    "
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
        ci=.95,
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