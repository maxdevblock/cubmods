"""
CUB models in Python.
Module for CUBSH (Combination of Uniform
and Binomial with Shelter Effect).

Description:
    This module contains methods and classes
    for CUBSH model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import cubsh

    samp = pd.read_csv("ordinal.csv")
    fit = cubsh.mle(samp.rv, m=7, sh=5s)
    print(fit.summary())
    fit.plot()
    plt.show()

References:
    * TODO: add references

List of TODO:
    * 

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
#import pandas as pd
from scipy.special import binom
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, probbit, dissimilarity,
    #conf_ell,
    #chisquared,
    InvalidCategoriesError,
    lsat, luni, aic, bic,
)
from . import cub
from .smry import CUBres, CUBsample

#TODO anytime a function is called, use explicit kwargs!!!
###################################################################
# FUNCTIONS
###################################################################

def pidelta_to_pi1pi2(pi, delta):
    pi1 = (1-delta)*pi
    pi2 = (1-delta)*(1-pi)
    return pi1, pi2

def pi1pi2_to_pidelta(pi1, pi2):
    pi = pi1/(pi1+pi2)
    delta = 1 - pi1 - pi2
    return pi, delta

def pidelta_to_lambdeta(pi, delta):
    lambd = pi*(1-delta)
    eta = ((1-pi)*(1-delta))/(1-pi*(1-delta))
    return lambd, eta

def lambdeta_to_pidelta(lambd, eta):
    pi = lambd/(lambd+eta*(1-lambd))
    delta = (1-lambd)*(1-eta)
    return pi, delta

def pmf_delta(m, sh, pi, xi, delta):
    """
    PMF of CUBSH model with delta
    """
    R = choices(m)
    D = R==sh
    #print(m, pi, xi, R)
    p = delta*D + (1-delta) * ( pi*binom(m-1, R-1) * (1-xi)**(R-1) * xi**(m-R) + (1-pi)/m )
    return p

def pmf(m, sh, pi1, pi2, xi):
    """
    PMF of CUBSH model with pi1, pi2
    """
    R = choices(m)
    D = (R==sh).astype(int)
    p = pi1*probbit(m, xi) + pi2/m + (1-pi1-pi2)*D
    return p

def proba(m, sh, pi1, pi2, xi, r):
    p = pmf(m, sh, pi1, pi2, xi)
    return p[r-1]

def proba_delta(m, sh, pi, xi, delta, r):
    """
    probability Pr(R=r) of CUBSH model
    with delta
    """
    #print(m, pi, xi, R)
    D = r==sh
    p = delta*D + (1-delta) * ( pi*binom(m-1, r-1) * (1-xi)**(r-1) * xi**(m-r) + (1-pi)/m )
    #print(p)
    return p

def cmf(m, sh, pi1, pi2, xi):
    return pmf(m, sh, pi1, pi2, xi).cumsum()

def cmf_delta(m, sh, pi, xi, delta):
    """
    CMF of CUBSH model with delta
    """
    return pmf(m, pi, xi, sh, delta).cumsum()

def mean_delta(m, sh, pi, xi, delta):
    """
    mean of CUBSH model with delta
    """
    mu = cub.mean(m, pi, xi)
    mi = mu + delta*(mu-sh)
    return mi

def var_delta(m, pi, xi, delta):
    """
    variance of CUBSH model with delta
    """
    v = ((1-delta)**2)*cub.var(m, pi, xi)
    return v

def std_delta(m, pi, xi, delta):
    """
    standard deviation of CUB model
    """
    s = np.sqrt(var_delta(m, pi, xi, delta))
    return s

#TODO: is it ok???
def skew(pi, xi):
    """
    skewness normalized eta index
    """
    return pi*(1/2-xi)

#TODO: test
def mean_diff(m, sh, pi1, pi2, xi):
    R = choices(m)
    S = choices(m)
    mu = 0
    for r in R:
        for s in S:
            mu += abs(r-s)*proba(m,sh,pi1,pi2,xi,r)*proba(m,sh,pi1,pi2,xi,s)
    return mu

#TODO: test
def median(m, sh, pi1, pi2, xi):
    R = choices(m)
    cp = cmf(m, sh, pi1, pi2, xi)
    M = R[cp>.5][0]
    if M > R.max():
        M = R.max()
    return M

#TODO: test
def gini(m, sh, pi1, pi2, xi):
    ssum = 0
    for r in choices(m):
        ssum += proba(m, sh, pi1, pi2, xi, r)**2
    return m*(1-ssum)/(m-1)

#TODO: test
def laakso(m, sh, pi1, pi2, xi):
    g = gini(m, sh, pi1, pi2, xi)
    return g/(m - (m-1)*g)

def loglik(m, sh, pi1, pi2, xi, f):
    L = pmf(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi)
    #TODO: check log invalid value from mle
    l = (f*np.log(L)).sum()
    return l

def varcov(m, sh, pi1, pi2, xi, n):
    """
    compute asymptotic variance-covariance
    of CUBSH estimated parameters
    controllare n!
    """
    R = choices(m)
    pr = pmf(m, sh, pi1, pi2, xi)
    dd = np.repeat(0, m)
    dd[sh-1] = 1
    bb = probbit(m, xi)

    aaa = bb-dd
    bbb = (1/m)-dd
    c4 = pi1*bb*(m-R-xi*(m-1))/(xi*(1-xi))
    atilde = aaa/pr
    btilde = bbb/pr
    ctilde = c4/pr

    d11 = np.sum(aaa*atilde)
    d22 = np.sum(bbb*btilde)
    dxx = np.sum(c4*ctilde)
    d12 = np.sum(bbb*atilde)
    d1x = np.sum(c4*atilde)
    d2x = np.sum(c4*btilde)

    #TODO: infmat in R style
    infmat = np.ndarray(shape=(3,3))
    infmat[0,0] = d11
    infmat[1,1] = d22
    infmat[2,2] = dxx
    infmat[0,1] = d12
    infmat[1,0] = d12
    infmat[0,2] = d1x
    infmat[2,0] = d1x
    infmat[1,2] = d2x
    infmat[2,1] = d2x

    varmat = np.ndarray(shape=(3,3))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)/n
    return varmat

def init_theta(f, m, sh):
    pi1, xi = cub.init_theta(f, m)
    fc = f[sh-1]/f.sum()
    #print("fc", fc)
    deltamax = (m*fc-1)/(m-1)
    delta = np.max([.01, deltamax])
    pi2 = np.max([.01, 1-delta-pi1])
    return pi1, pi2, xi

def plot_simplex(pi1pi2list, ax=None, fname=None):
    # tick length
    tkl = .02
    # tick step
    steps = .05
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    # simplex axes
    ax.plot([-1, 0], [0, 0], "k")
    ax.plot([-1, -.5], [0, np.sin(np.pi/3)], "k")
    ax.plot([-.5, 0], [np.sin(np.pi/3), 0], "k")

    sin = np.sin(np.pi/3)
    cos = np.cos(np.pi/3)
    for tick in np.arange(0, 1, steps)[1:]:
        ### grid
        # delta grid
        ax.plot(
            [-1+tick/2, -tick/2],
            [tick*sin, tick*sin],
            "k", alpha=.1
        )
        # delta ticks
        ax.plot(
            [-1+tick/2-tkl, -1+tick/2],
            [tick*sin, tick*sin], "k"
        )
        # pi1 grid
        ax.plot(
            [-tick, -tick-(1-tick)*cos],
            [0, (1-tick)*sin],
            "k", alpha=.1
        )
        # pi1 ticks
        ax.plot(
            [-tick+tkl*cos, -tick],
            [-tkl*sin, 0], "k"
        )
        # pi2 grid
        ax.plot(
            [-tick, -(tick)*cos],
            [0, (tick)*sin],
            "k", alpha=.1
        )
        # pi2 ticks
        ax.plot(
            [-tick*cos, -(tick-tkl)*cos],
            [tick*sin, (tick+tkl)*sin], "k"
        )
        ### axes
        # pi1
        ax.text(
            -tick+tkl*cos+tkl/2, -tkl*sin,
            f" {tick:.2f}", ha="center", va="top",
            rotation=300
        )
        # delta
        ax.text(
            -1+tick/2-tkl, tick*sin,
            f"{tick:.2f} ", ha="right", va="center"
        )
        # pi2
        ax.text(
            -(tick-tkl/2)*cos,
            (tick+2*tkl)*sin+tkl/4,
            f" {1-tick:.2f}", ha="left", va="center",
            rotation=60
        )
    ### labels
    ax.text(
        -1+.5/2-5*tkl, .5*sin,
        r"$\delta$", ha="right", va="center",
        fontsize=20
    )
    ax.text(
        -.5, -4.5*tkl,
        r"$\pi_1$", ha="center", va="top",
        fontsize=20
    )
    ax.text(
        -(.5-tkl)*cos+6.5*tkl,
        .5*sin,
        r"$\pi_2$", ha="right", va="center",
        fontsize=20
    )
    # no axes and equal
    plt.axis('off')
    ax.set_aspect('equal', 'box')
    # points
    for i, pi1pi2 in enumerate(pi1pi2list):
        pi1 = pi1pi2[0]
        pi2 = pi1pi2[1]
        _, delta = pi1pi2_to_pidelta(pi1, pi2)
        x = -pi1-delta*cos
        y = delta*sin
        ax.scatter(x, y, facecolor="None",
            edgecolor="k", s=200)
        ax.text(x, y, f"{chr(i+65)}", ha="center", va="center")
    if ax is None:
        fig.tight_layout()
    if fname:
        fig.savefig(fname)
    return ax

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, sh, pi1, pi2, xi, n, seed=None):
    """
    generate random sample from CUBSH model
    from pi1 and pi2
    """
    if m<= 4:
        print("ERR: Number of ordered categories should be at least 5")
        raise InvalidCategoriesError(m=m, model="cubsh")
    np.random.seed(seed)
    theoric = pmf(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=theoric
        )
    pi, delta = pi1pi2_to_pidelta(pi1,pi2)
    pars = np.array([pi1, pi2, xi,
        pi, delta])
    par_names = np.array([
        "pi1", "pi2", "xi",
        "*pi", "*delta"
    ])
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    sample = CUBsample(
        model="CUBSH",
        rv=rv, m=m, sh=sh,
        pars=pars, par_names=par_names,
        theoric=theoric, diss=diss
    )
    return sample

def draw2(m, sh, pi, xi, delta, n, seed=None):
    """
    generate random sample from CUBSH model
    from pi and delta
    """
    pi1, pi2 = pidelta_to_pi1pi2(pi, delta)
    sample = draw(m, sh, pi1, pi2, xi, n, seed=seed)
    return sample

###################################################################
# INFERENCE
###################################################################

def mle(sample, m, sh, maxiter=500, tol=1e-4,
    gen_pars=None
    ):
    if m<= 4:
        print("ERR: Number of ordered categories should be at least 5")
        raise InvalidCategoriesError(m=m, model="cubsh")

    start = dt.datetime.now()
    R = choices(m)
    f = freq(sample=sample, m=m)
    n = sample.size
    #aver = np.mean(sample) #TODO: unused aver?
    #varcamp = np.mean(sample**2)-aver**2 #TODO: unused?
    dd = (R==sh).astype(int)
    pi1, pi2, xi = init_theta(f=f, m=m, sh=sh)
    l = loglik(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi, f=f)
    niter = 1
    while niter <= maxiter:
        lold = l
        bb = probbit(m=m, xi=xi)
        tau1 = pi1*bb
        tau2 = pi2/m
        denom = tau1+tau2+(1-pi1-pi2)*dd
        tau1 /= denom
        tau2 /= denom
        #TODO: unused tau3...?
        #tau3 = 1-tau1-tau2
        numaver = np.sum(R*f*tau1)
        denaver = np.sum(f*tau1)
        averpo = numaver/denaver
        # updated estimates
        pi1 = np.sum(f*tau1)/n
        pi2 = np.sum(f*tau2)/n
        xi = (m-averpo)/(m-1)
        if xi < .001:
            xi = .001
            niter = maxiter-1
        l = loglik(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi, f=f)
        lnew = l
        testll = np.abs(lnew-lold)
        if testll < tol:
            break
        else:
            l = lnew
        niter += 1

    if xi > .999: xi = .99
    if xi < .001: xi = .01
    if pi1 < .001: pi1 = .01

    varmat = varcov(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi, n=n)
    end = dt.datetime.now()
    durata = (end-start).total_seconds()
    pi, delta = pi1pi2_to_pidelta(pi1=pi1, pi2=pi2)

    esdelta = np.sqrt(varmat[0,0]+varmat[1,1]+2*varmat[0,1])
    walddelta = delta/esdelta
    pvaldelta = np.round(2*abs(sps.norm.sf(walddelta)), 20)

    espi = np.sqrt((pi1**2)*varmat[1,1] + (pi2**2)*varmat[0,0] - 2*pi1*pi2*varmat[0,1])/((pi1+pi2)**2)
    waldpi = pi/espi
    pvalpi = np.round(2*abs(sps.norm.sf(waldpi)), 20)

    trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + 3*np.log(trvarmat/3) - np.log(np.linalg.det(varmat))

    stime = np.array([pi1, pi2, xi])
    errstd = np.sqrt(np.diag(varmat))
    wald = stime/errstd
    pval = np.round(2*abs(sps.norm.sf(wald)), 20)
    
    estimates = np.concatenate((
        [pi1, pi2, xi],
        [pi, xi, delta]
    ))
    est_names = np.array([
        "pi1", "pi2", "xi",
        "pi", "xi", "delta"
    ])
    e_types = np.array([
        "Alternative parametrization",
        None, None,
        "Uncertainty", "Feeling",
        "Shelter effect"
    ])
    stderrs = np.concatenate((
        errstd, [espi], [errstd[-1]],
        [esdelta]
    ))
    wald = np.concatenate((
        wald, [waldpi], [wald[-1]],
        [walddelta]
    ))
    pval = np.concatenate((
        pval, [pvalpi], [pval[-1]],
        [pvaldelta]
    ))

    theoric = pmf(m=m, sh=sh, pi1=pi1, pi2=pi2, xi=xi)
    diss = dissimilarity(f/n, theoric)
    loglikuni = luni(m=m, n=n)
    #xisb = (m-aver)/(m-1) #TODO: unused?
    #llsb = cub.loglik(m, 1, xisb, f) #TODO: unused?
    #TODO: use nonzero in lsat?
    #nonzero = np.nonzero(f)
    logliksat = lsat(f=f, n=n)
    # mean loglikelihood
    muloglik = l/n
    # deviance from saturated model
    dev = 2*(logliksat-l)

    #pearson = (f-n*theorpr)/np.sqrt(n*theorpr) #TODO: unused?
    #X2 = np.sum(pearson**2) #TODO: unused?
    #relares = (f/n-theorpr)/theorpr #TODO: unused?

    #LL2 = 1/(1+np.mean((f/(n*theoric)-1)**2))
    #ll2 = (l-llunif)/(logsat-llunif)
    # FF2 is the overall fraction of correct responses, as predicted by the estimated model
    #FF2 = 1-dissim
    AIC = aic(l=l, p=3)
    BIC = bic(l=l, p=3, n=n)
    #TODO: compute Corr(pi,xi)
    # rho = varmat[0,1]/np.sqrt(varmat[0,0]*varmat[1,1])
    #rho = None

    return CUBresCUBSH(
        model="CUBSH",
        m=m, sh=sh, n=n,
        niter=niter, maxiter=maxiter,
        tol=tol, theoric=theoric,
        estimates=estimates,
        est_names=est_names,
        e_types=e_types,
        stderrs=stderrs,
        wald=wald, pval=pval,
        loglike=l, loglikuni=loglikuni,
        logliksat=logliksat,
        muloglik=muloglik, dev=dev,
        AIC=AIC, BIC=BIC,
        seconds=durata, time_exe=start,
        sample=sample, f=f, varmat=varmat,
        diss=diss,
        gen_pars=gen_pars,
    )

class CUBresCUBSH(CUBres):

    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        #pi1 = self.estimates[0]
        #pi2 = self.estimates[1]
        pi = self.estimates[3]
        xi = self.estimates[4]
        delta = self.estimates[5]
        title = "CUBSH model    "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$ , $\delta={delta:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        #TODO: add diss_gen
        # if self.diss_gen is not None:
        #     title += "\n"
        #     title += fr"Gener($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
        ax.set_title(title)

        R = choices(self.m)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability mass")

        #p = pmf(m=self.m, pi1=self.pi1, pi2=self.pi2, xi=self.xi, sh=self.sh)
        ax.plot(R, self.theoric, ".b:",
            label="estimated", ms=10)
        ax.scatter(R, self.f/self.n,
            facecolor="None",
            edgecolor="k", s=200,
            label="observed")
        if self.gen_pars is not None:
            pi1_gen, pi2_gen = self.gen_pars["pi1"], self.gen_pars["pi2"]
            xi_gen = self.gen_pars['xi']
            p_gen = pmf(m=self.m, pi1=pi1_gen, pi2=pi2_gen, xi=xi_gen, sh=self.sh)
            ax.stem(R, p_gen, linefmt="--r",
            markerfmt="none", label="generating")

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

    #TODO: add displacement from CUB with no shelter effect
    def plot_confell(self,
        figsize=(7, 5),
        ci=.95,
        equal=True,
        magnified=False,
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )

        if equal:
            ax.set_aspect("equal")
        #TODO: compute Corr(pi,xi)
        # ax.set_title(f"Corr(pi,xi)= {self.rho}")
        pi = self.estimates[3]
        xi = self.estimates[4]
        delta = self.estimates[5]
        ax.set_xlabel(r"$(1-\pi)$  uncertainty")
        ax.set_ylabel(r"$(1-\xi)$  preference")

        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)

        ax.plot(1-pi, 1-xi, 
            ".b",ms=20, alpha=.5,
            label="estimated")
        ax.text(1-pi, 1-xi,
            fr"  $\delta = {delta:.3f}$" "\n",
            ha="left", va="bottom")
        if self.gen_pars is not None:
            pass
            #TODO: add gen_pars
            #ax.scatter(1-self.pi_gen, 1-self.xi_gen,
            #    facecolor="None",
            #    edgecolor="r", s=200, label="generating")

        #alpha = 1 - ci
        #z = abs(sps.norm().ppf(alpha/2))
        # Horizontal CI
        #ax.plot(
        #    [1-(self.pi-z*self.espi),
        #    1-(self.pi+z*self.espi)],
        #    [1-self.xi, 1-self.xi],
        #    "b", lw=1
        #)
        # Vertical CI
        #ax.plot(
        #    [1-self.pi, 1-self.pi],
        #    [1-(self.xi-z*self.stderrs[1]),
        #    1-(self.xi+z*self.stderrs[1])],
        #    "b", lw=1
        #)
        #TODO: from SIGMA{pi1,pi2,delta} to SIGMA{pi,xi}
        # Confidence Ellipse
        # conf_ell(
        #     self.varmat,
        #     1-self.pi, 1-self.xi,
        #     ci, ax
        # )

        if not magnified:
            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            ticks = np.arange(0, 1.05, .1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if xlim[0] < 0:
                ax.set_xlim((0, xlim[1]))
                xlim = ax.get_xlim()
            if xlim[1] > 1:
                ax.set_xlim((xlim[0], 1))
            if ylim[0] < 0:
                ax.set_ylim((0, ylim[1]))
                ylim = ax.get_ylim()
            if ylim[1] > 1:
                ax.set_ylim((ylim[0], 1))
            # ax.axline(
            #     [1-self.pi, 1-self.xi],
            #     slope=self.rho, ls="--"
            # )

        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        ax.grid(visible=True)

        if ax is None:
            if saveas is not None:
                fig.savefig(saveas,
                    bbox_inches='tight')
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
        plot CUBSH model fitted from a sample
        """
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        self.plot_ordinal(ax=ax[0])
        self.plot_confell(ci=ci, ax=ax[1])
        pi1 = self.estimates[0]
        pi2 = self.estimates[1]
        # self.plot_confell(
        #     ci=ci, ax=ax[2],
        #     magnified=True, equal=False)
        plot_simplex([(pi1, pi2)], ax=ax[2])
        plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
