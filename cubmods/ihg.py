# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
"""
CUB models in Python.
Module for IHG (Inverse HyperGeometric).

Description:
============
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
import scipy.stats as sps
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_hess
#import matplotlib.pyplot as plt
from .general import (
    choices, freq, aic, bic,
    luni, lsat, dissimilarity
)
from .smry import CUBres, CUBsample

def pmf(m, theta):
    pr = np.repeat(np.nan, m)
    pr[0] = theta
    for i in range(m-1):
        j = i + 1
        pr[j] = pr[i]*(1-theta)*(m-j)/(m-j-1+j*theta)
    return pr

def loglik(m, theta, f):
    p = pmf(m=m, theta=theta)
    l = (f*np.log(p)).sum()
    return l

def effe(theta, m, f):
    return -loglik(m=m, theta=theta, f=f)

def init_theta(m, f):
    R = choices(m)
    aver = (R*f).sum()/f.sum()
    est = (m-aver)/(1+(m-2)*aver)
    return est

def draw(m, theta, n, seed=None):
    """
    generate random sample from CUB model
    """
    theoric = pmf(m=m, theta=theta)
    np.random.seed(seed)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=theoric
    )
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    pars = np.array([theta])
    par_names = np.array(["theta"])
    sample = CUBsample(
        model="IHG",
        rv=rv, m=m,
        pars=pars,
        par_names=par_names,
        seed=seed, theoric=theoric,
        diss=diss
    )
    return sample

def var(m, theta):
    n = theta*(1-theta)*(m-theta)*(m-1)**2
    d1 = (theta*(m-2)+1)**2
    d2 = (theta*(m-3)+2)
    return n/(d1*d2)

def mle(m, sample, gen_pars=None):
    start = dt.datetime.now()
    f = freq(sample=sample, m=m)
    n = sample.size
    theta0 = init_theta(m, f)
    opt = minimize(
        effe, x0=theta0,
        #bracket=(0, 1),
        bounds=[(1e-16, 1-1e-16)],
        args=(m, f),
        method="L-BFGS-B",
    )
    theta = opt.x
    infmat = approx_hess([theta0], effe,
        args=(m, f))
    #varmat = np.ndarray(shape=(opt.size,opt.size))
    varmat = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    end = dt.datetime.now()
    stderrs = np.sqrt(np.diag(varmat))
    wald = theta/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    #print(theta, stderrs, wald, pval)
    l = loglik(m=m, theta=theta, f=f)
    muloglik = l/n
    AIC = aic(l=l, p=1)
    BIC = bic(l=l, p=1, n=n)
    loglikuni = luni(m=m, n=n)
    logliksat = lsat(n=n, f=f)
    dev = 2*(logliksat-l)
    theoric = pmf(m=m, theta=theta)
    diss = dissimilarity(f/n, theoric)
    #end = dt.datetime.now()
    
    #print(f"theta={theta}")
    #print(f"SE={stderrs}")
    
    return CUBresIHG(
        model="IHG",
        m=m, n=n,
        theoric=theoric,
        e_types=["Theta"],
        est_names=["theta"],
        estimates=theta,
        stderrs=stderrs, pval=pval,
        wald=wald, loglike=l,
        muloglik=muloglik,
        logliksat=logliksat,
        loglikuni=loglikuni,
        dev=dev, AIC=AIC, BIC=BIC,
        diss=diss, sample=sample,
        f=f, varmat=varmat,
        seconds=(end-start).total_seconds(),
        time_exe=start, gen_pars=gen_pars
    )

class CUBresIHG(CUBres):
    
    def plot_estim(self, ci=.95, ax=None,
        magnified=False):
        theta = self.estimates
        se = self.stderrs
        # change all spines
        for axis in ['bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)
        ax.plot(theta, 0, ".b",
            ms=20, alpha=.5,
            label="estimated")
        if ci is not None:
            alpha = 1-ci
            z = abs(sps.norm().ppf(alpha/2))
            ax.plot(
                [theta-z*se, theta+z*se],
                [0, 0],
                "b", lw=1
            )
        if self.gen_pars is not None:
            ax.scatter(self.gen_pars['theta'], 0,
                facecolor="None",
                edgecolor="r", s=200, label="generator")
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        ax.set_yticks([])
        ax.grid(True)
        if not magnified:
            ax.set_xlim([0,1])
            ax.set_xticks(np.arange(
                0,10.1)/10)
        ax.set_xlabel(r"$\theta$ parameter")
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        title = f"{self.model} model    "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\theta={self.estimates[0]:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        if self.gen_pars is not None:
            title += "\n"
            title += fr"Gener($\theta={self.gen_pars['theta']:.3f}$)"
            p_gen = pmf(m=self.m, theta=self.gen_pars['theta'])
            R = choices(m=self.m)
            ax.stem(R, p_gen, linefmt="--r",
                markerfmt="none", label="generating")
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
        ci=.95,
        saveas=None,
        figsize=(7, 15)
        ):
        """
        plot CUB model fitted from a sample
        """
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        self.plot_ordinal(ax=ax[0])
        self.plot_estim(ax=ax[1], ci=ci)
        self.plot_estim(ax=ax[2], ci=ci,
            magnified=True)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
