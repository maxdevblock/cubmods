"""
CUB models in Python.
Module for CUBSH2 (Combination of Uniform
and Binomial with two Shelter Choices).

Description:
============
    This module contains methods and classes
    for CUBSH2 model family.
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
  TODO: check plots

@Author:      Massimo Pierini
@Institution: Università degli Studi di Brescia
@Affiliation: Analytics and Data Science for Economics and Management
@Date:        2024
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements
#import pickle
import datetime as dt
import numpy as np
#import pandas as pd
from scipy.special import binom
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, probbit, dissimilarity,
    conf_ell, plot_ellipsoid,
    #chisquared,
    InvalidCategoriesError,
    lsat, luni, aic, bic,
)
from . import cub
from . import cubsh
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def pdz_to_p123(pi, delta, zeta):
    """
    pi1, pi2, pi3 from
    pi, delta, zeta
    """
    pi1 = (1-delta)*pi
    pi2 = (1-delta)*(1-pi)
    pi3 = delta*zeta
    return pi1, pi2, pi3

def p123_to_pdz(pi1, pi2, pi3):
    """
    pi, delta, zeta from
    pi1, pi2, pi3
    """
    pi = pi1/(pi1+pi2)
    delta = 1 - pi1 - pi2
    zeta = pi3/delta
    return pi, delta, zeta

def pmf(m, sh1, sh2, pi1, pi2, pi3, xi):
    """
    PMF of CUBSH2 model with pi1, pi2
    """
    R = choices(m)
    D1 = (R==sh1).astype(int)
    D2 = (R==sh2).astype(int)
    p = (pi1*probbit(m, xi)
        + pi2/m + pi3*D1
        + (1-pi1-pi2-pi3)*D2)
    return p

def prob(m, sh1, sh2, pi1, pi2, pi3, xi, r):
    p = pmf(m, sh1, sh2, pi1, pi2, pi3, xi)
    return p[r-1]

def cmf(m, sh1, sh2, pi1, pi2, pi3, xi):
    return pmf(m, sh1, sh2, pi1, pi2, pi3, xi).cumsum()

def mean(m, sh1, sh2, pi1, pi2, pi3, xi):
    """
    mean of CUBSH2 model
    """
    pi, delta, zeta = p123_to_pdz(pi1,pi2,pi3)
    mu = cub.mean(m, pi, xi)
    mi = (delta*(zeta*sh1 + (1-zeta)*sh2)
        + (1-delta)*mu
    )
    return mi

def var(m, pi1, pi2, pi3, xi):
    """
    variance of CUBSH2 model
    """
    pi, delta, _ = p123_to_pdz(pi1,pi2,pi3)
    v = ((1-delta)**2)*cub.var(m, pi, xi)
    return v

def std(m, pi1, pi2, pi3, xi):
    """
    standard deviation of CUBSH2 model
    """
    s = np.sqrt(var(
        m, pi1, pi2, pi3, xi))
    return s

#TODO: skew
def skew(pi, xi):
    """
    skewness normalized eta index
    """
    return None #pi*(1/2-xi)

#TODO: test mean_diff
def mean_diff(m, sh1, sh2, pi1, pi2, pi3, xi):
    R = choices(m)
    S = choices(m)
    mu = 0
    for r in R:
        for s in S:
            mu += abs(r-s)*prob(m,sh1,sh2,pi1,pi2,pi3,xi,r)*prob(m,sh1,sh2,pi1,pi2,pi3,xi,s)
    return mu

#TODO: test median
def median(m, sh1, sh2, pi1, pi2, pi3, xi):
    R = choices(m)
    cp = cmf(m, sh1, sh2, pi1, pi2, pi3, xi)
    M = R[cp>.5][0]
    if M > R.max():
        M = R.max()
    return M

#TODO: test gini
def gini(m, sh1, sh2, pi1, pi2, pi3, xi):
    ssum = 0
    for r in choices(m):
        ssum += prob(m, sh1, sh2, pi1, pi2, pi3, xi, r)**2
    return m*(1-ssum)/(m-1)

#TODO: test laakso
def laakso(m, sh1, sh2, pi1, pi2, pi3, xi):
    g = gini(m, sh1, sh2, pi1, pi2, pi3, xi)
    return g/(m - (m-1)*g)

def loglik(m, sh1, sh2, pi1, pi2, pi3, xi, f):
    L = pmf(m=m, sh1=sh1, sh2=sh2,
        pi1=pi1, pi2=pi2, pi3=pi3, xi=xi)
    #TODO: check log invalid value from mle
    l = (f*np.log(L)).sum()
    return l

def varcov(m, sh1, sh2, pi1, pi2, pi3, xi, n):
    """
    compute asymptotic variance-covariance
    of CUBSH2 parameters estimators
    """
    R = choices(m)
    #f = freq(sample=sa)
    pr = pmf(m, sh1, sh2, pi1, pi2, pi3, xi)
    D1 = (R==sh1).astype(int)
    D2 = (R==sh2).astype(int)
    bb = probbit(m, xi)
    uu = 1/m

    bbb = pi1*bb*(m-R - xi*(m-1))/(xi*(1-xi))

    # infmat structure
    # | pi1pi1 | pi1pi2 | pi1pi3 | pi1xi |
    # | pi2pi1 | pi2pi2 | pi2pi3 | pi2xi |
    # | pi3pi1 | pi3pi2 | pi3pi3 | pi3xi |
    # | xi pi1 | xi pi2 | xi pi3 | xi xi |

    d11 = np.sum((bb-D2)**2/pr)
    d12 = np.sum((bb-D2)*(uu-D2)/pr)
    d13 = np.sum((bb-D2)*(D1-D2)/pr)
    d14 = np.sum((bb-D2)*bbb/pr)
    d22 = np.sum((uu-D2)**2/pr)
    d23 = np.sum((uu-D2)*(D1-D2)/pr)
    d24 = np.sum((uu-D2)*bbb/pr)
    d33 = np.sum((D1-D2)**2/pr)
    d34 = np.sum((D1-D2)*bbb/pr)
    d44 = np.sum((bbb**2)/pr)

    #TODO: infmat in R style?
    infmat = np.ndarray(shape=(4,4))
    infmat[0,0] = d11
    infmat[0,1] = d12
    infmat[0,2] = d13
    infmat[0,3] = d14
    infmat[1,0] = d12
    infmat[1,1] = d22
    infmat[1,2] = d23
    infmat[1,3] = d24
    infmat[2,0] = d13
    infmat[2,1] = d23
    infmat[2,2] = d33
    infmat[2,3] = d34
    infmat[3,0] = d14
    infmat[3,1] = d24
    infmat[3,2] = d34
    infmat[3,3] = d44
    #print(infmat)

    varmat = np.ndarray(shape=(4,4))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)/n
    return varmat

def init_theta(f, m, sh1, sh2,
    verbose=False):
    """
    Initialize parameters for EM
    algorithm
    """
    pi1, xi = cub.init_theta(f, m)
    fc1 = f[sh1-1]/f.sum()
    fc2 = f[sh2-1]/f.sum()
    d1 = (fc1*(m-1)+fc2-1)/(m-2)
    pi3 = max([.01, d1])
    d2 = (fc2*(m-1)+fc1-1)/(m-2)
    pi4 = max([.01, d2])
    pi2 = 1-pi1-pi3-pi4
    if verbose:
        print("#########")
        print("init")
        print(f"pi1 = {pi1:.3f}")
        print(f"pi2 = {pi2:.3f}")
        print(f"pi3 = {pi3:.3f}")
        print(f"xi  = {xi:.3f}")
    return pi1, pi2, pi3, xi

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, sh1, sh2, pi1, pi2, pi3, xi,
    n, seed=None):
    """
    generate random sample from CUBSH model
    from pi1 and pi2
    """
    if m<= 5:
        print("ERR: Number of ordered categories should be at least 5")
        raise InvalidCategoriesError(m=m, model="cubsh2")
    np.random.seed(seed)
    theoric = pmf(m=m, sh1=sh1, sh2=sh2,
        pi1=pi1, pi2=pi2, pi3=pi3, xi=xi)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=theoric
        )
    pi, delta, zeta = p123_to_pdz(
        pi1, pi2, pi3)
    pars = np.array([pi1, pi2, pi3, xi,
        pi, delta, zeta])
    par_names = np.array([
        "pi1", "pi2", "pi3", "xi",
        "*pi", "*delta", "*zeta"
    ])
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    sample = CUBsample(
        model="CUBSH2",
        rv=rv, m=m, sh=[sh1, sh2],
        pars=pars, par_names=par_names,
        theoric=theoric, diss=diss
    )
    return sample

#TODO: test draw2
def draw2(m, sh1, sh2, pi, xi, delta, zeta, n, seed=None):
    """
    generate random sample from CUBSH2 model
    from pi, delta, and zeta
    """
    pi1, pi2, pi3 = pdz_to_p123(pi, delta, zeta)
    sample = draw(m, sh1, sh2, pi1, pi2, pi3, xi, n, seed=seed)
    return sample

###################################################################
# INFERENCE
###################################################################

def mle(sample, m, sh1, sh2,
    maxiter=1000, tol=1e-4,
    gen_pars=None
    ):
    """
    Maximum likelihood estimation of parameters pi1, pi2, pi3, xi
    with EM algorithm
    """
    if m<= 5:
        print("ERR: Number of ordered categories should be at least 5")
        raise InvalidCategoriesError(m=m, model="cubsh2")
    if sh1 == sh2:
        print("ERR: c1=c2, use cubsh instead")
        raise Exception("Invalid shelter choices")

    start = dt.datetime.now()
    R = choices(m)
    f = freq(sample=sample, m=m)
    n = sample.size
    dd1 = (R==sh1).astype(int)
    dd2 = (R==sh2).astype(int)
    pi1, pi2, pi3, xi = init_theta(f=f, m=m, sh1=sh1, sh2=sh2)
    l = loglik(m=m, sh1=sh1, sh2=sh2,
        pi1=pi1, pi2=pi2, pi3=pi3, xi=xi,
        f=f)
    niter = 1
    while niter <= maxiter:
        lold = l
        bb = probbit(m=m, xi=xi)
        tau1 = pi1*bb
        tau2 = pi2/m
        tau3 = pi3*dd1
        tau4 = (1-pi1-pi2-pi3)*dd2
        denom = tau1+tau2+tau3+tau4
        tau1 /= denom
        tau2 /= denom
        tau3 /= denom
        tau4 /= denom
        taut = tau1+tau2+tau3+tau4
        pi41 = 1 - pi2 - pi3
        pi42 = 1 - pi1 - pi3
        pi43 = 1 - pi1 - pi2
        #tau3 = 1-tau1-tau2
        numaver = np.sum(R*f*tau1)
        denaver = np.sum(f*tau1)
        averpo = numaver/denaver
        # updated estimates
        pi1 = (f*tau1).sum()/n
        pi2 = (f*tau2).sum()/n
        pi3 = (f*tau3).sum()/n
        xi = (m-averpo)/(m-1)
        if xi < .001:
            xi = .001
            niter = maxiter-1
        l = loglik(m=m, sh1=sh1, sh2=sh2,
            pi1=pi1, pi2=pi2, pi3=pi3,
            xi=xi, f=f)
        lnew = l
        testll = np.abs(lnew-lold)
        if testll < tol:
            break
        else:
            l = lnew
        niter += 1

    #return (pi1, pi2, pi3, xi)

    if xi > .999: xi = .99
    if xi < .001: xi = .01
    if pi1 < .001: pi1 = .01
    #TODO: other adjustments to final estimates?

    varmat = varcov(m=m, sh1=sh1, sh2=sh2,
        pi1=pi1, pi2=pi2, pi3=pi3, xi=xi,
        n=n)
    end = dt.datetime.now()
    durata = (end-start).total_seconds()

    pi4 = 1 - pi1 - pi2 - pi3
    espi4 = np.sqrt(
        varmat[0,0] + varmat[1,1] +
        varmat[2,2] + 2*varmat[0,1]
        + 2*varmat[0,2] + 2*varmat[1,2]
    )
    waldpi4 = pi4 / espi4
    pvalpi4 = np.round(2*abs(sps.norm.sf(waldpi4)), 20)

    pi, delta, zeta = p123_to_pdz(
        pi1, pi2, pi3)

    espi = np.sqrt(
        pi2**2 * varmat[0,0]
        +
        pi1**2 * varmat[1,1]
        -
        2*pi1*pi2 * varmat[0,1]
    )/( (pi1+pi2)**2 )
    waldpi = pi / espi
    pvalpi = np.round(2*abs(sps.norm.sf(waldpi)), 20)

    esdelta = np.sqrt(
        varmat[0,0] + varmat[1,1]
        + 2*varmat[0,1]
    )
    walddelta = delta / esdelta
    pvaldelta = np.round(2*abs(sps.norm.sf(walddelta)), 20)

    eszeta = np.sqrt(
        (1-pi1-pi2)**2 * varmat[2,2] +
        pi3**2 * varmat[0,0] +
        pi3**2 * varmat[1,1] +
        2*(1-pi1-pi2)*pi3 * varmat[0,2] +
        2*(1-pi1-pi2)*pi3 * varmat[1,2] +
        2*pi3**2 * varmat[0,1]
    ) / ((1-pi1-pi2)**2)
    waldzeta = zeta / eszeta
    pvalzeta = np.round(2*abs(sps.norm.sf(waldzeta)), 20)

    stime = np.array([pi1, pi2, pi3, xi])
    errstd = np.sqrt(np.diag(varmat))
    wald = stime/errstd
    pval = np.round(2*abs(sps.norm.sf(wald)), 20)

    estimates = np.concatenate((
        [pi1, pi2, pi3, xi, pi4],
        [pi, xi, delta, zeta]
    ))
    est_names = np.array([
        "pi1", "pi2", "pi3", "xi", "(pi4)",
        "pi", "xi", "delta", "zeta"
    ])
    e_types = np.array([
        "Alternative parametrization",
        None, None, None, None,
        "Uncertainty", "Feeling",
        "Shelter effects", "1st Shelter Choice"
    ])
    stderrs = np.concatenate((
        errstd, [espi4],
        [espi], [errstd[-1]],
        [esdelta], [eszeta]
    ))
    wald = np.concatenate((
        wald, [waldpi4],
        [waldpi], [wald[-1]],
        [walddelta], [waldzeta]
    ))
    pval = np.concatenate((
        pval, [pvalpi4],
        [pvalpi], [pval[-1]],
        [pvaldelta], [pvalzeta]
    ))

    theoric = pmf(m=m, sh1=sh1, sh2=sh2,
        pi1=pi1, pi2=pi2, pi3=pi3, xi=xi)
    diss = dissimilarity(f/n, theoric)
    loglikuni = luni(m=m, n=n)
    #xisb = (m-aver)/(m-1)
    #llsb = cub.loglik(m, 1, xisb, f)
    #TODO: use nonzero in lsat?
    #nonzero = np.nonzero(f)
    logliksat = lsat(f=f, n=n)
    # mean loglikelihood
    muloglik = l/n
    # deviance from saturated model
    dev = 2*(logliksat-l)

    #pearson = (f-n*theorpr)/np.sqrt(n*theorpr)
    #X2 = np.sum(pearson**2)
    #relares = (f/n-theorpr)/theorpr

    #LL2 = 1/(1+np.mean((f/(n*theoric)-1)**2))
    #ll2 = (l-llunif)/(logsat-llunif)
    # FF2 is the overall fraction of correct responses, as predicted by the estimated model
    #FF2 = 1-dissim
    AIC = aic(l=l, p=4)
    BIC = bic(l=l, p=4, n=n)

    return CUBresCUBSH2(
        model="CUBSH2",
        m=m, sh=[sh1,sh2], n=n,
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

def varcov_pixi(n, m,
    sh1, sh2, pi1, pi2, pi3, xi,
    debug=False):
    """
    Variance-covariance of (pi,xi) estimated parameters
    """
    R = choices(m=m)
    br = probbit(m=m, xi=xi)
    ur = 1/m
    pi, delta, _ = p123_to_pdz(
        pi1=pi1, pi2=pi2, pi3=pi3)
    pr = pmf(m=m, sh1=sh1, sh2=sh2,
        pi1=pi1, pi2=pi2, pi3=pi3,
        xi=xi)
    p_pi = (1-delta)*(br-ur)
    p_xi = (1-delta)*pi*br*(m-R-xi*(m-1))/(xi*(1-xi))
    d11 = n*(p_pi**2/pr).sum()
    d12 = n*(p_pi*p_xi/pr).sum()
    d22 = n*(p_xi**2/pr).sum()
    infmat = np.ndarray(shape=(2,2))
    infmat[0,0] = d11
    infmat[0,1] = d12
    infmat[1,0] = d12
    infmat[1,1] = d22

    varmat = np.ndarray(shape=(2,2))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
        if debug:
            print(f"es(pi)={np.sqrt(varmat[0,0]):.3f}")
            print(f"es(xi)={np.sqrt(varmat[1,1]):.3f}")
    return varmat

class CUBresCUBSH2(CUBres):

    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None, fig=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        pi1 = self.estimates[0]
        pi2 = self.estimates[1]
        pi3 = self.estimates[2]
        xi = self.estimates[3]
        genpi1 = None
        genpi2 = None
        genpi3 = None
        genxi  = None
        #pi = self.estimates[3]
        #xi = self.estimates[4]
        #delta = self.estimates[5]
        title = "CUBSH2 model    "
        title += f"$n={self.n}$ , "
        title += f"$m={self.m}$ , "
        title += f"$c_1={self.sh[0]}$ , "
        title += f"$c_2={self.sh[1]}$\n"
        title += fr"Estim($\pi_1={pi1:.3f}$ , $\pi_2={pi2:.3f}$ , $\pi_3={pi3:.3f}$ , $\xi={xi:.3f}$)"
        title += f"\nDissim(est,obs)={self.diss:.4f}"
        if self.gen_pars is not None:
            #pass
            genpi1 = self.gen_pars['pi1']
            genpi2 = self.gen_pars['pi2']
            genpi3 = self.gen_pars['pi3']
            genxi  = self.gen_pars['xi']
            title += "\n"
            title += fr"Gener($\pi_1={genpi1:.3f}$ , "
            title += fr"$\pi_2={genpi2:.3f}$ , "
            title += fr"$\pi_3={genpi3:.3f}$ , "
            title += fr"$\xi={genxi:.3f}$ , "
            
        #TODO: add diss_gen
        # if self.diss_gen is not None:
        #     title += "\n"
        #     title += fr"Gener($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
        ax.set_title(title)

        R = choices(self.m)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability")

        #p = pmf(m=self.m, pi1=self.pi1, pi2=self.pi2, xi=self.xi, sh=self.sh)
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
        if self.gen_pars is not None:
            p_gen = pmf(m=self.m,
                pi1=genpi1, pi2=genpi2,
                pi3=genpi3, xi=genxi,
                sh1=self.sh[0],
                sh2=self.sh[1])
            ax.stem(R, p_gen, linefmt="--r",
                markerfmt="none", label="generating")

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        #if ax is None:
        if saveas is not None:
            print(f"Saving plot as `{saveas}`")
            fig.savefig(saveas, bbox_inches='tight')
        else:
            return fig, ax
        #else:
            #return ax

    #TODO: add displacement from CUB with no shelter effect
    def plot_confell(self,
        figsize=(7, 5),
        ci=.95,
        equal=True,
        magnified=False,
        ax=None,
        saveas=None,
        confell=False,
        debug=False,
        cubdisp=True
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )

        if equal:
            ax.set_aspect("equal")

        pi = self.estimates[5]
        xi = self.estimates[6]
        #delta = self.estimates[5]
        ax.set_xlabel(r"$(1-\pi)$  uncertainty")
        ax.set_ylabel(r"$(1-\xi)$  feeling")

        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)

        ax.plot(1-pi, 1-xi, 
            ".b",ms=20, alpha=.5,
            mfc="none",
            label="estimated")
        #ax.text(1-pi, 1-xi,
        #    fr"  $\delta = {delta:.3f}$" "\n",
        #    ha="left", va="bottom")
        if self.gen_pars is not None:
            gpi1 = self.gen_pars["pi1"]
            gpi2 = self.gen_pars["pi2"]
            gpi3 = self.gen_pars["pi3"]
            gxi = self.gen_pars["xi"]
            gpi,_,_ = p123_to_pdz(
                gpi1, gpi2, gpi3
            )
            ax.plot(1-gpi, 1-gxi, 
                "*r",ms=9, alpha=.5,
                mfc="none",
                label="generating")

        # Confidence Ellipse
        if confell:
            Vpx = varcov_pixi(n=self.n,
                m=self.m, sh1=self.sh[0],
                sh2=self.sh[1],
                pi1=self.estimates[0],
                pi2=self.estimates[1],
                pi3=self.estimates[2],
                xi=self.estimates[3])

            conf_ell(
                 Vpx,
                 1-pi, 1-xi,
                 ci, ax
            )

        if cubdisp and not magnified:
            cubest = cub.mle(
                sample=self.sample,
                m=self.m,
            )
            cubpi = cubest.estimates[0]
            cubxi = cubest.estimates[1]
            ax.plot(1-cubpi, 1-cubxi,
                "sg", ms=9, alpha=.75,
                mfc="none",
                label="CUB model")
            ax.annotate(
                "", (1-pi, 1-xi),
                (1-cubpi, 1-cubxi),
                arrowprops=dict(
                    facecolor="black",
                    arrowstyle="-",
                    #alpha=.25,
                    #shrink=.05,
                    #width=1,
                    zorder=np.inf
                )
            )

            for shi in [0,1]:
                cubshest = cubsh.mle(
                    sample=self.sample,
                    m=self.m, sh=self.sh[shi]
                )
                cubshpi = cubshest.estimates[0]
                cubshxi = cubshest.estimates[1]
                ax.plot(1-cubshpi, 1-cubshxi,
                    "s", ms=9, alpha=.75,
                    mfc="none", color=f"C{shi+3}",
                    label=f"CUBSH $c={self.sh[shi]}$")
                ax.annotate(
                    "", (1-pi, 1-xi),
                    (1-cubshpi, 1-cubshxi),
                    arrowprops=dict(
                        facecolor="black",
                        arrowstyle="-",
                        #alpha=.25,
                        #shrink=.05,
                        #width=1,
                        zorder=np.inf
                    )
                )

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

    def plot3d(self, ax, ci=.95,
        magnified=False):
        pi = self.estimates[3]
        xi = self.estimates[4]
        de = self.estimates[5]
        V = varcov_pxd(
            self.m, self.sh, pi, xi,
            de, self.n)
        #print()
        #print("VARCOV(pxd)")
        #print(V)
        #espxd = np.sqrt(
        #            np.diag(V))
        #print()
        #print("ES(pxd)")
        #print(espxd)
        plot_ellipsoid(V=V,
            E=(1-pi,1-xi,de), ax=ax,
            zlabel=r"Shelter Choice $\delta$",
            magnified=magnified, ci=ci
        )

    def plot(self,
        ci=.95,
        saveas=None,
        confell=False,
        debug=False,
        test3=False,
        figsize=(7, 15)
        ):
        """
        plot CUBSH model fitted from a sample
        """
        fig, ax = plt.subplots(3, 1,
            figsize=figsize,
            constrained_layout=True)
        self.plot_ordinal(ax=ax[0])
        if test3:
            ax[1].remove()
            ax[2].remove()
            ax[1] = fig.add_subplot(3,1,2,
                projection='3d')
            ax[2] = fig.add_subplot(3,1,3,
                projection='3d')
            self.plot3d(ax=ax[1], ci=ci)
            self.plot3d(ax=ax[2], ci=ci,
                magnified=True)
        else:
            self.plot_confell(ci=ci, ax=ax[1],
                confell=confell, debug=debug)
            self.plot_confell(ci=ci, ax=ax[2],
                confell=confell, debug=debug,
                magnified=True)
            #pi1 = self.estimates[0]
            #pi2 = self.estimates[1]
            # self.plot_confell(
            #     ci=ci, ax=ax[2],
            #     magnified=True, equal=False)
            #plot_simplex([(pi1, pi2)], ax=ax[2])
            #plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
