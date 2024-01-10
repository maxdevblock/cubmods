import datetime as dt
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_hess
from .general import (
    logis, colsof, choices, freq,
    aic, bic, lsat, luni, dissimilarity,
    lsatcov
)
from .cush2 import pmf as pmf_cush2
from .smry import CUBres

def pmf(m, sh1, sh2,
    omega1, omega2,
    X1, X2):
    d1 = logis(X1, omega1)
    d2 = logis(X2, omega2)
    n = X1.shape[0]
    p_i = np.ndarray(shape=(n,m))
    for i in range(n):
        p_i[i] = pmf_cush2(m=m, c1=sh1,
            c2=sh2, d1=d1[i], d2=d2[i])
    p = p_i.mean(axis=0)
    return p

def loglik(sample, m, sh1, sh2,
    omega1, omega2,
    X1, X2):
    delta1 = logis(X1, omega1)
    delta2 = logis(X2, omega2)
    D1 = (sample==sh1).astype(int)
    D2 = (sample==sh2).astype(int)
    l = np.sum(np.log(
        delta1*D1 + delta2*D2 +
        (1-delta1-delta2)/m
    ))
    return l

def effe(pars, sample, m, sh1, sh2, X1, X2):
    w1 = colsof(X1)+1
    omega1 = pars[:w1]
    omega2 = pars[w1:]
    l = loglik(sample, m, sh1, sh2,
        omega1, omega2,
        X1, X2)
    return -l

def mle(sample, m, sh1, sh2,
    X1, X2, gen_pars=None,
    maxiter=None, tol=None):
    start = dt.datetime.now()
    w1 = colsof(X1)
    w2 = colsof(X2)
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
    delta2_0 = max([
        .01, (fc2*(m-1)+fc1-1)/(m-2)])
    om2_0 = np.log(delta2_0/(1-delta2_0))
    om2 = np.concatenate((
        [om2_0], np.repeat(.1, w2)
    ))
    pars = np.concatenate((om1, om2))
    optim = minimize(
        effe, x0=pars,
        args=(sample, m, sh1, sh2, X1, X2)
    )
    estimates = optim.x
    omega1 = estimates[:(w1+1)]
    omega2 = estimates[(w1+1):]
    est_names = np.concatenate((
        ["constant"],
        [x for x in X1.columns],
        ["constant"],
        [x for x in X2.columns],
    ))
    e_types = np.concatenate((
        ["Shelter effect 1"],
        [None for _ in X1.columns],
        ["Shelter effect 2"],
        [None for _ in X2.columns],
    ))
    
    infmat = approx_hess(estimates, effe,
        args=(sample, m, sh1, sh2, X1, X2))
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
    stderrs = np.sqrt(np.diag(varmat))
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    l = loglik(m=m, sample=sample,
        omega1=omega1, omega2=omega2,
        sh1=sh1, sh2=sh2, X1=X1, X2=X2)
    logliksat = lsat(m=m, n=n, f=f)
    logliksatcov = lsatcov(
        sample=sample,
        covars=[X1, X2]
    )
    loglikuni = luni(m=m, n=n)
    dev = 2*(logliksat-l)
    theoric = pmf(m=m, sh1=sh1, sh2=sh2,
        omega1=omega1, omega2=omega2,
        X1=X1, X2=X2)
    diss = dissimilarity(f/n, theoric)
    muloglik = l/n
    AIC = aic(l=l, p=estimates.size)
    BIC = bic(l=l, p=estimates.size, n=n)
    end = dt.datetime.now()
    
    return CUBresCUSH2XX(
        model="2CUSH(XX)",
        m=m, n=n, sh=np.array([sh1, sh2]),
        estimates=estimates,
        est_names=est_names,
        e_types=e_types,
        stderrs=stderrs, pval=pval,
        theoric=theoric,
        wald=wald, loglike=l,
        muloglik=muloglik,
        loglikuni=loglikuni,
        logliksat=logliksat,
        logliksatcov=logliksatcov,
        dev=dev, AIC=AIC, BIC=BIC,
        sample=sample, f=f, varmat=varmat,
        diss=diss,
        seconds=(end-start).total_seconds(),
        time_exe=start
    )

class CUBresCUSH2XX(CUBres):
    
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