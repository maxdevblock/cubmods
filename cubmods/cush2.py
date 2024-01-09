import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as sps
#from scipy.optimize import minimize
#import seaborn as sns
import matplotlib.pyplot as plt
from .general import (
    conf_ell, freq, dissimilarity,
    choices, aic, bic, luni, lsat,
    NoShelterError
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

def draw(m, sh, delta1, delta2, n, seed=None):
    """
    generate random sample from CUB model
    """
    if sh is None:
        raise NoShelterError(model="cush2")
    c1 = sh[0]; c2 = sh[1]
    if delta1+delta2 > 1:
        raise Exception("delta1+delta2>1")
    theoric = pmf(m, c1, c2, delta1, delta2)
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
        sh=sh, pars=pars,
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
    logliksat = lsat(m=m, n=n, f=f)
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
        time_exe=start
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
