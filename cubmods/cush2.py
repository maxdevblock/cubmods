"""
CUB models in Python.
Module for 2-CUSH (Combination of Uniform
and 2 Shelter Choices).

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
    """
    Probability mass of 2-CUSH model,
    Combination of Uniform and 2
    Shelter Choices.
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

def draw(m, sh1, sh2,
    delta1, delta2, n, seed=None):
    """
    generate random sample from CUB model
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
    #sh=np.array([c1, c2])
    sample = CUBsample(
        model="2CUSH",
        rv=rv, m=m,
        sh=np.array([sh1, sh2]),
        pars=pars,
        par_names=par_names,
        seed=seed, theoric=theoric,
        diss=diss
    )
    return sample

def varcov(m, n, d1, d2, fc1, fc2):
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

def mle(sample, m, c1, c2, gen_pars=None):
    """
    Maximum Likelihood Estimation of
    delta1 and delta2 parameters in
    a 2-CUSH model.
    """
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
        model="2CUSH",
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
        gen_pars=gen_pars
    )

def loglik(sample, m, c1, c2):
    #l = (f*np.log(pr(m, d1, d2))).sum()
    n = sample.size
    fc1 = (sample==c1).sum()/n
    fc2 = (sample==c2).sum()/n
    fc3 = 1-fc1-fc2
    l = n*(fc1*np.log(fc1) + 
        fc2*np.log(fc2) + 
        fc3*np.log(fc3/(m-2)))
    return l

def effe(d, sample, m, c1, c2):
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

    def plot_par_space(self,
        figsize=(7, 5),
        ax=None, ci=.95,
        saveas=None):

        estd1, estd2 = self.estimates
        c1, c2 = self.sh

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )

        if self.gen_pars is not None:
            d1 = self.gen_pars["delta1"]
            d2 = self.gen_pars["delta2"]
            ax.plot(d1, d2, "xr",
                label="generating",
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
        ax.set_title("2-CUSH model parameter space")
        if ax is None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
            else:
                return fig, ax
        else:
            return ax
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        estd1, estd2 = self.estimates
        title = f"{self.model} model "
        title += fr"($c_1={self.sh[0]}$ , $c_2={self.sh[1]}$)"
        title += f"    $n={self.n}$\n"
        title += fr"Estim($\delta_1={estd1:.3f}$ , $\delta_2={estd2:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        if self.gen_pars is not None:
            title += "\n"
            title += fr"Gener($\delta_1={self.gen_pars['delta1']:.3f}$ , $\delta_2={self.gen_pars['delta2']:.3f}$)"
            p_gen = pmf(c1=self.sh[0], c2=self.sh[1], d1=estd1, d2=estd2, m=self.m)
            R = choices(m=self.m)
            ax.stem(R, p_gen, linefmt="--r",
                markerfmt="none", label="generating")
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
        ci=.95,
        saveas=None,
        figsize=(7, 11)
        ):
        """
        plot CUB model fitted from a sample
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
