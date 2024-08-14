# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
"""
CUB models in Python.
Module for IHG (Inverse HyperGeometric) with covariates.

Description:
    This module contains methods and classes
    for IHG model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import ihg

    samp = pd.read_csv("ordinal.csv")
    fit = ihg.mle(samp.rv, m=7)
    print(fit.summary())
    fit.plot()
    plt.show()

...
References:
===========
  - TODO: aggiungere tesi?
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
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_hess
from .general import (
    logis, freq, choices, aic, bic,
    #lsat, 
    luni, dissimilarity,
    #lsatcov, 
    addones, colsof,
)
from .ihg import pmf as pmf_ihg
from .smry import CUBres, CUBsample

def pmfi(m, V, nu):
    n = V.shape[0]
    p_i = np.ndarray(shape=(n,m))
    theta = logis(V, nu)
    for i in range(n):
        p_i[i] = pmf_ihg(m=m, theta=theta[i])
    return p_i

def pmf(m, V, nu):
    """
    Test pmf
    """
    p = pmfi(m, V, nu).mean(axis=0)
    return p

def draw(m, nu, V, 
    df, orig_df, formula, seed=None):
    n = V.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    R = choices(m)
    p = pmfi(m, V, nu)
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
        V.columns
    ))
    
    return CUBsample(
        model="IHG(V)",
        m=m,
        pars=nu,
        par_names=par_names,
        theoric=theoric,
        diss=diss,
        df=orig_df, formula=formula,
        rv=rv.astype(int),
        seed=seed
    )

def probi(m, sample, V, nu):
    n = sample.size
    theta = logis(V, nu)
    p = np.repeat(np.nan, n)
    for i in range(n):
        prob = pmf_ihg(m=m, theta=theta[i])
        p[i] = prob[sample[i]-1]
    return p

def loglik(m, sample, V, nu):
    p = probi(m, sample, V, nu)
    l = np.sum(np.log(p))
    return l

def effe(nu, m, sample, V):
    l = loglik(m, sample, V, nu)
    return -l

def init_theta(m, f):
    R = choices(m)
    aver = np.sum(f*R)/np.sum(f)
    est = (m-aver)/(1+(m-2)*aver)
    return est

def mle(m, sample, V,
    df, formula, gen_pars=None):
    start = dt.datetime.now()
    f = freq(m=m, sample=sample)
    n = sample.size
    theta0 = init_theta(m, f)
    #VV = addones(V)
    v = colsof(V)
    nu0 = np.log(theta0/(1-theta0))
    nuini = np.concatenate((
        [nu0], np.repeat(.1, v)
    ))
    optim = minimize(
        effe, x0=nuini,
        args=(m, sample, V),
        #method="Nelder-Mead"
    )
    nu = optim.x
    l = loglik(m, sample, V, nu)
    infmat = approx_hess(nu, effe,
        args=(m, sample, V))
    varmat = np.ndarray(shape=(nu.size,nu.size))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.array(nu)
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    l = loglik(m=m, sample=sample, nu=nu,
        V=V)
    #logliksat = lsat(n=n, f=f)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[V]
    #)
    loglikuni = luni(m=m, n=n)
    muloglik = l/n
    #dev = 2*(logliksat-l)
    AIC = aic(l=l, p=estimates.size)
    BIC = bic(l=l, p=estimates.size, n=n)
    theoric = pmf(m=m, nu=nu, V=V)
    diss = dissimilarity(f/n, theoric)
    est_names = np.concatenate((
        ["constant"],
        V.columns
    ))
    #print(est_names.shape)
    e_types = np.concatenate((
        ["Theta"],
        [None for _ in V.columns]
    ))
    
    end = dt.datetime.now()

    return CUBresIHGV(
        model="IHG(V)",
        m=m, n=n,
        theoric=theoric,
        e_types=e_types,
        est_names=est_names,
        estimates=estimates,
        stderrs=stderrs,
        pval=pval,
        wald=wald, loglike=l,
        muloglik=muloglik,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        #dev=dev,
        AIC=AIC, BIC=BIC,
        diss=diss, sample=sample,
        f=f, varmat=varmat,
        seconds=(end-start).total_seconds(),
        time_exe=start, gen_pars=gen_pars,
        df=df, formula=formula
    )

class CUBresIHGV(CUBres):
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
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