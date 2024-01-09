#TODO: implement
import datetime as dt
import pickle
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
from statsmodels.tools.numdiff import approx_hess
import matplotlib.pyplot as plt
from .general import (
    logis, freq, dissimilarity,
    aic, bic, lsat, luni, choices
)
from .cush import pmf as pmf_cush
from .smry import CUBres

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

class CUSHsample(object): #TODO: armonizzare
    def __init__(self, rv, m, omega, n, X, sh, seed=None):
        self.m = m
        self.omega = omega
        self.rv = rv
        self.n  = n
        self.X = X
        self.sh = sh
        self.seed = seed

    def __str__(self):
        return f"CUBsample(m={self.m}, sh={self.sh}, omega={self.omega}, n={self.n})"

    def summary(self):
        diss = dissimilarity(
            freq(self.rv, self.m)/self.n,
            pmf(m=self.m, sh=self.sh, omega=self.omega, X=self.X)
        )
        smry = "=======================================================================\n"
        smry += "=====>>> CUB  model    <<<=====   Generated random sample\n"
        smry += "=======================================================================\n"
        smry += f"m={self.m}  Sample size={self.n}  sh={self.sh}  omega={self.omega}  seed={self.seed}\n"
        smry += "=======================================================================\n"
        smry += "Shelter Effect"
        #smry += f"(1-pi) = {1-self.pi:.6f}\n"
        #smry += "Feeling\n"
        #smry += f"(1-xi) = {1-self.xi:.6f}\n"
        smry += "=======================================================================\n"
        smry += f"Mean      = {np.mean(self.rv):.6f}\n"
        smry += f"Variance  = {np.var(self.rv, ddof=1):.6f}\n"
        smry += f"Std. Dev. = {np.std(self.rv, ddof=1):.6f}\n"
        smry += f"-----------------------------------------------------------------------\n"
        smry += f"Dissimilarity =  {diss:.7f}\n"
        smry += "======================================================================="
        return smry

    def plot(self, figsize=(7, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        R = choices(self.m)
        f = freq(self.rv, self.m)
        ax.scatter(R, f/self.rv.size, facecolor="None",
            edgecolor="k", s=200, label="generated")
        p = pmf(m=self.m, sh=self.sh, omega=self.omega, X=self.X)
        ax.stem(R, p, linefmt="--r",
            markerfmt="none", label="theoric")
        ax.set_xticks(R)
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_xlabel("Options")
        ax.set_ylabel("Probability mass")
        ax.set_title(self)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        return fig

    def save(self, fname):
        """
        Save a CUBsample object to file
        """
        filename = f"{fname}.cub.sample"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Sample saved to {filename}")

def draw(m, sh, omega, X, seed=None):
    n = X.shape[0]
    R = choices(m)
    p = pmfi(m, sh, omega, X)
    rv = np.repeat(np.nan, n)
    for i in range(n):
        np.random.seed(seed)
        rv[i] = np.random.choice(
            R,
            size=1, replace=True,
            p=p[i]
        )
    return CUSHsample(
        m=m, sh=sh, omega=omega, X=X,
        rv=rv, n=n, seed=seed
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
    XX = np.c_[np.ones(X.shape[0]), X]
    x = XX.shape[1] - 1
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
    infmat = approx_hess(omi, effe,
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
    logliksat = lsat(m=m, f=f, n=n)
    dev = 2*(logliksat-l)
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
        logliksat=logliksat,
        dev=dev, AIC=AIC, BIC=BIC,
        sample=sample, f=f, varmat=varmat,
        X=X, diss=diss,
        seconds=(end-start).total_seconds(),
        time_exe=start
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
