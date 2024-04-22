"""
CUB models in Python.
Module for CUSH (Combination of Uniform
and Shelter effect).

Description:
    This module contains methods and classes
    for CUSH model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import cush

    samp = pd.read_csv("ordinal.csv")
    fit = cush.mle(samp.rv, m=7, sh=5)
    print(fit.summary())
    fit.plot()
    plt.show()

References:
    * Capecchi, S., & Piccolo, D. (2017).
      Dealing with heterogeneity in ordinal responses.
      Quality & Quantity, 51, 2375-2393.
      DOI: 10.1007/s11135-016-0393-3

List of TODOs:
    * TODO: check gini & laakso

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements
import pickle
import datetime as dt
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, dissimilarity,
    chisquared, lsat, luni, aic, bic,
    NoShelterError
)
from . import cub
from .smry import CUBres, CUBsample

#TODO anytime a function is called, use explicit kwargs!!!
###################################################################
# FUNCTIONS
###################################################################

def pmf(m, sh, delta):
    R = choices(m=m)
    s = (R==sh).astype(int)
    p = delta*(s-1/m)+1/m
    return p

def loglik(sample, m, sh, delta):
    n = sample.size
    f = freq(sample=sample, m=m)
    fc = f[sh-1]/n
    l = n*((1-fc)*np.log(1-delta)
        +fc*np.log(1+(m-1)*delta)
        -np.log(m)
    )
    return l

def mean(m, sh, delta):
    mu = delta*sh+(1-delta)*(m+1)/2
    return mu

def var(m, sh, delta):
    va = (1-delta)*(delta*(sh-(m+1)/2)**2+(m**2-1)/12)
    return va

def gini(delta):
    return 1-delta**2

def laakso(m, delta):
    l = (1-delta**2)/(1+(m-1)*delta**2)
    return l

def LRT(m, fc, n):
    """
    Returns lambda of LRT
    """
    a = fc*np.log(fc)
    b = (1-fc)*np.log((1-fc)/(m-1))
    c = np.log(m)
    return 2*n*(a+b+c)

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, sh, delta, n, seed=None):
    """
    generate random sample from CUB model
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
        diss=diss
    )
    return sample

###################################################################
# INFERENCE
###################################################################

def mle(sample, m, sh,
    gen_pars=None,
    maxiter=None, tol=None #for GEM compatibility
    ):
    _, _ = maxiter, tol
    if sh is None:
        raise NoShelterError(model="cush")
    start = dt.datetime.now()
    f = freq(sample=sample, m=m)
    n = sample.size
    aver = np.mean(sample)
    fc = f[sh-1]/n
    deltaest = np.max([.01, (m*fc-1)/(m-1)])
    #TODO: check stderr
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
    #TODO: what is xisb?
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
    gen_pars=gen_pars,
    )

class CUBresCUSH(CUBres):

    #TODO add options to plot:
    #     * bars (default)
    #     * confidence ellipse
    #     * magnified confidence ellipse
    def plot(self,
        ci=.95,
        saveas=None,
        figsize=(7, 15)
        ):
        """
        plot CUB model fitted from a sample
        """
        R = choices(self.m)
        #print(R, self.f, self.n)
        delta = self.estimates[0]
        title = fr"$n={self.n}$    "
        title += fr"estim($\delta={delta:.3f}$)"
        title += f"\nDissim(est,obs)={self.diss:.4f}"
        X2 = None

        fig, ax = plt.subplots(3, 1, figsize=figsize)
        ax[0].set_xticks(R)
        ax[0].set_xlabel("Ordinal")
        ax[0].set_ylabel("Probability mass")
        ax[1].set_xlim((0,1))
        #ax[1].set_ylim((0,1))
        ticks = np.arange(0, 1.1, .1)
        ax[1].set_xticks(ticks)
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[1].set_xlabel(r"$\delta$  shelter effect")
        #ax[1].set_ylabel(r"$(1-\xi)$  preference")
        ax[2].set_xlabel(r"$\delta$  shelter effect")
        #ax[2].set_ylabel(r"$(1-\xi)$  preference")

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
        if self.gen_pars is not None:
            delta_gen = self.gen_pars["delta"]
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

#TODO: pearson, X2, relares, FF2, LL2, II2, AIC, BIC, ICOMP in general?