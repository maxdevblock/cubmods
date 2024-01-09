"""
CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial).

Description:
    This module contains methods and classes
    for CUB model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import cush

    samp = pd.read_csv("ordinal.csv")
    fit = cub.mle(samp.rv, m=7)
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
import datetime as dt
import pickle
import numpy as np
from scipy.special import binom
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, dissimilarity,
    conf_ell, probbit,
    InvalidCategoriesError,
    ParameterOurOfBoundsError,
    InvalidSampleSizeError,
    #chisquared,
)
from .smry import CUBres, CUBsample

#TODO anytime a function is called, use explicit kwargs!!!
###################################################################
# FUNCTIONS
###################################################################

def pmf(m, pi, xi):
    """
    PMF of CUB model
    """
    R = choices(m)
    #print(m, pi, xi, R)
    p = pi*binom(m-1, R-1) * (1-xi)**(R-1) * xi**(m-R) + (1-pi)/m
    return p

def proba(m, pi, xi, r):
    """
    probability Pr(R=r) of CUB model
    """
    #print(m, pi, xi, R)
    p = pi*binom(m-1, r-1) * (1-xi)**(r-1) * xi**(m-r) + (1-pi)/m
    #print(p)
    return p

def cmf(m, pi, xi):
    """
    CMF of CUB model
    """
    return pmf(m, pi, xi).cumsum()

def mean(m, pi, xi):
    """
    mean of CUB model
    """
    return (m+1)/2 + pi*(m-1)*(1/2-xi)

def var(m, pi, xi):
    """
    variance of CUB model
    """
    v =  (m-1)*(pi*xi*(1-xi) + (1-pi)*((m+1)/12+pi*(m-1)*(xi-1/2)**2))
    return v

def std(m, pi, xi):
    """
    standard deviation of CUB model
    """
    return np.sqrt(var(m, pi, xi))

def skew(pi, xi):
    """
    skewness normalized eta index
    """
    return pi*(1/2-xi)

def mean_diff(m, pi, xi):
    R = choices(m)
    S = choices(m)
    mu = 0
    for r in R:
        for s in S:
            mu += abs(r-s)*proba(m,pi,xi,r)*proba(m,pi,xi,s)
    return mu

def median(m, pi, xi):
    R = choices(m)
    cp = cmf(m, pi, xi)
    M = R[cp>.5][0]
    if M > R.max():
        M = R.max()
    return M

def gini(m, pi, xi):
    ssum = 0
    for r in choices(m):
        ssum += proba(m, pi, xi, r)**2
    return m*(1-ssum)/(m-1)

def laakso(m, pi, xi):
    g = gini(m, pi, xi)
    return g/(m - (m-1)*g)

def rvs(m, pi, xi, n):
    """
    generate random sample from CUB model
    """
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=pmf(m, pi, xi)
        )
    return rv

def loglik(m, pi, xi, f):
    L = pmf(m, pi, xi)
    l = (f*np.log(L)).sum()
    return l

def varcov(m, pi, xi, ordinal):
    """
    compute asymptotic variance-covariance
    of CUB estimated parameters
    """
    #R = choices(m)
    # OLD WAY TO COMPUTE INFORMATION MATRIX
    # # Pr(R=r|pi=1,xi)
    # qr = pmf(m, 1, xi)
    # # Pr(R=r|pi,xi)
    # pr = pmf(m, pi, xi)
    # dpr_dpi = qr-1/m
    # dpr_dxi = pi*qr*(m-xi*(m-1)-R)/(xi*(1-xi))

    vvi = (m-ordinal)/xi-(ordinal-1)/(1-xi)
    ui = (m-ordinal)/(xi**2)+(ordinal-1)/((1-xi)**2)
    pri = pmf(m=m, pi=pi, xi=xi)
    qi = 1/(m*pri[ordinal-1])
    qistar = 1-(1-pi)*qi
    qitilde = qistar*(1-qistar)
    i11 = np.sum((1-qi)**2)/(pi**2)
    i12 =  -np.sum(vvi*qi*qistar)/pi
    i22 = np.sum(qistar*ui-(vvi**2)*qitilde)

    infmat = np.ndarray(shape=(2,2))
    # OLD WAY TO COMPUTE INFORMATION MATRIX
    # infmat[0,0] = np.sum(dpr_dpi**2/pr)
    # infmat[1,1] = np.sum(dpr_dxi**2/pr)
    # infmat[0,1] = np.sum(dpr_dpi*dpr_dxi/pr)
    # infmat[1,0] = infmat[0,1]
    #TODO: create matrix from array in R style
    #      matinf <- matrix(c(i11,i12,i12,i22), nrow=2, byrow=T)
    infmat[0,0] = i11
    infmat[1,1] = i22
    infmat[0,1] = i12
    infmat[1,0] = i12
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

def init_theta(f, m):
    #pi = .5
    #xi = (m-avg)/(m-1)
    F = f/f.sum()
    xi = 1 + (.5 - (np.argmax(F)+1))/m
    ppp = probbit(m, xi)
    pi = np.sqrt( (np.sum(F**2)-1/m) / (np.sum(ppp**2)-1/m) )
    pi = min([pi, .99])
    return pi, xi

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, pi, xi, n, seed=None):
    """
    generate random sample from CUB model
    """
    if m<= 3:
        print("ERR: Number of ordered categories should be at least 4")
        raise InvalidCategoriesError(m=m, model="cub")
    if xi < 0 or xi > 1:
        raise ParameterOurOfBoundsError("xi", xi)
    if pi < 0 or pi > 1:
        raise ParameterOurOfBoundsError("pi", pi)
    if n <= 0:
        raise InvalidSampleSizeError(n)

    np.random.seed(seed)
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=pmf(m=m, pi=pi, xi=xi)
        )
    pars = np.array([pi, xi])
    par_names = np.array(["pi", "xi"])
    theoric = pmf(m=m, xi=xi, pi=pi)
    f = freq(m=m, sample=rv)
    diss = dissimilarity(f/n, theoric)
    sample = CUBsample(
        model="CUB",
        rv=rv, m=m, pars=pars,
        par_names=par_names,
        theoric=theoric, diss=diss,
        seed=seed
    )
    return sample

###################################################################
# INFERENCE
###################################################################

def mle(sample, m,
    gen_pars=None,
    maxiter=500,
    tol=1e-4,
    #ci=.99
    ): #TODO: use ci for conf int in summary?
    """
    fit a sample to a CUB model
    with m preference choices.
    if the sample has been generated
    from a CUB model itself and
    generating (pi, xi) are known,
    compute compare metrics
    """
    if m<= 3:
        print("ERR: Number of ordered categories should be at least 4")
        raise InvalidCategoriesError(m=m, model="cub")
    # validate parameters
    #if not validate_pars(m=m, n=sample.size):
    #    pass
    # start datetime
    start = dt.datetime.now()
    # cast sample to numpy array
    sample = np.array(sample)
    # model preference choices
    R = choices(m)
    # observed absolute frequecies
    f = freq(sample, m)
    # sample size
    n = sample.size

    # initialize (pi, xi)
    pi, xi = init_theta(f, m)
    # compute loglikelihood
    l = loglik(m, pi, xi, f)

    # start E-M algorithm
    niter = 1
    while niter < maxiter:
        lold = l
        # pmf of shifted binomial
        sb = pmf(m, 1, xi)
        # posterior probabilities
        tau = 1/(1+(1-pi)/(m*pi*sb))
        ftau = f*tau
        # expected posterior probability
        Rnp = np.dot(R, ftau)/ftau.sum()
        # estimates of (pi, xi)
        pi = np.dot(f, tau)/n
        xi = (m-Rnp)/(m-1)
        # avoid division by zero
        if xi < .001:
            xi = .001
            niter = maxiter-1
        # new lohlikelihood
        l = loglik(m, pi, xi, f)
        lnew = l
        # compute delta-loglik
        deltal = abs(lnew-lold)
        # check tolerance
        if deltal <= tol:
            break
        else:
            l = lnew
        niter += 1
    # end E-M algorithm
    
    # avoid division by zero
    if xi >.999:
        xi = .99
    if xi < .001:
        xi = .01
    if pi < .001:
        pi = .01
    # variance-covariance matrix
    varmat = varcov(m=m, pi=pi, xi=xi, ordinal=sample)
    end = dt.datetime.now()
    # standard errors
    stderrs = np.array([
        np.sqrt(varmat[0,0]),
        np.sqrt(varmat[1,1])
    ])
    # Wald statistics
    wald = np.array([pi, xi])/stderrs
    # p-value
    pval = 2*(sps.norm().sf(abs(wald)))
    # Akaike Information Criterion
    AIC = -2*l + 2*(2)
    # Bayesian Information Criterion
    BIC = -2*l + np.log(n)*(2)
    # mean loglikelihood
    muloglik = l/n
    # loglik of null model (uniform)
    loglikuni = -(n*np.log(m))
    # loglik of saturated model
    logliksat = -(n*np.log(n)) + np.sum((f[f!=0])*np.log(f[f!=0]))
    # loglik of shiftet binomial
    xibin = (m-sample.mean())/(m-1)
    loglikbin = loglik(m, 1, xibin, f)
    # Explicative powers
    #Ebin = (loglikbin-loglikuni)/(logliksat-loglikuni)
    #Ecub = (l-loglikbin)/(logliksat-loglikuni)
    #Ecub0 = (l-loglikuni)/(logliksat-loglikuni)
    # deviance from saturated model
    dev = 2*(logliksat-l)
    # ICOMP metrics
    npars = 2
    trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + npars*np.log(trvarmat/npars) - np.log(np.linalg.det(varmat))
    # coefficient of correlation
    rho = varmat[0,1]/np.sqrt(varmat[0,0]*varmat[1,1])
    theoric = pmf(m=m, pi=pi, xi=xi)
    diss = dissimilarity(f/n, theoric)
    estimates = [pi, xi]
    est_names = ["pi", "xi"]
    e_types = ["Uncertainty", "Feeling"]
    # compare with known (pi, xi)
    #diss_gen = None
    #if pi_gen is not None and xi_gen is not None:
    #    p_gen = pmf(m=m, pi=pi_gen, xi=xi_gen)
    #    diss_gen = dissimilarity(p, p_gen)
    # results object
    res = CUBresCUB00(
            model="CUB00",
            m=m, n=n, niter=niter,
            maxiter=maxiter, tol=tol,
            theoric=theoric,
            estimates=estimates,
            est_names=est_names,
            e_types=e_types,
            stderrs=stderrs,
            pval=pval, wald=wald,
            loglike=l, muloglik=muloglik,
            loglikuni=loglikuni,
            logliksat=logliksat,
            #loglikbin=loglikbin,
            #Ebin=Ebin, Ecub=Ecub, Ecub0=Ecub0,
            dev=dev, AIC=AIC, BIC=BIC,
            #ICOMP=ICOMP, 
            seconds=(end-start).total_seconds(),
            time_exe=start,
            rho=rho,
            sample=sample, f=f,
            varmat=varmat,
            diss=diss,
            #diss_gen=diss_gen,
            gen_pars=gen_pars
            #pi_gen=pi_gen, xi_gen=xi_gen
        )
    return res

class CUBresCUB00(CUBres):

    #TODO: add to cube, cush and cubsh
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        pi = self.estimates[0]
        xi = self.estimates[1]
        title = "CUB model    "
        title += f"$n={self.n}$\n"
        title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
        if self.gen_pars is not None:
            title += "\n"
            title += fr"Gener($\pi={self.gen_pars['pi']:.3f}$ , $\xi={self.gen_pars['xi']:.3f}$)"
            #title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
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
        if self.gen_pars is not None:
            p_gen = pmf(self.m, self.gen_pars['pi'], self.gen_pars['xi'])
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

    #TODO: add to cube, cush and cubsh???
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
        ax.set_title(f"Corr(pi,xi)= {self.rho:.4f}")

        pi = self.estimates[0]
        xi = self.estimates[1]

        ax.set_xlabel(r"$(1-\pi)$  uncertainty")
        ax.set_ylabel(r"$(1-\xi)$  preference")

        ax.plot(1-pi, 1-xi,
            ".b",ms=20, alpha=.5,
            label="estimated")
        if self.gen_pars is not None:
            ax.scatter(1-self.gen_pars['pi'], 1-self.gen_pars['xi'],
                facecolor="None",
                edgecolor="r", s=200, label="generating")

        alpha = 1 - ci
        z = abs(sps.norm().ppf(alpha/2))
        # # Horizontal CI
        # ax.plot(
        #     [1-(self.pi-z*self.stderrs[0]),
        #     1-(self.pi+z*self.stderrs[0])],
        #     [1-self.xi, 1-self.xi],
        #     "b", lw=1
        # )
        # # Vertical CI
        # ax.plot(
        #     [1-self.pi, 1-self.pi],
        #     [1-(self.xi-z*self.stderrs[1]),
        #     1-(self.xi+z*self.stderrs[1])],
        #     "b", lw=1
        # )
        # Confidence Ellipse
        conf_ell(
            self.varmat,
            1-pi, 1-xi,
            ci, ax
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
            # beta1 = self.varmat[0,1] / self.varmat[0,0]
            # ax.axline(
            #     [1-self.pi, 1-self.xi],
            #     slope=beta1, ls="--"
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
        plot CUB model fitted from a sample
        """
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        self.plot_ordinal(ax=ax[0])
        self.plot_confell(ci=ci, ax=ax[1])
        self.plot_confell(
            ci=ci, ax=ax[2],
            magnified=True, equal=False)
        plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax

    def save(self, fname):
        """
        Save a CUBresult object to file
        """
        filename = f"{fname}.cub.fit"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Fitting saved to {filename}")
