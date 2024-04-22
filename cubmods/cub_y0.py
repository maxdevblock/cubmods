"""
CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial) with covariates.

Description:
    This module contains methods and classes
    for CUB_Y0 model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    TODO: add example

References:
    TODO: add references

List of TODOs:
    * 

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
from .general import (
    logis, bitxi, probbit, choices,
    freq, hadprod, 
    #lsat, 
    luni,
    dissimilarity, aic, bic,
    colsof, addones
)
from .cub import (
    init_theta, pmf as cub_pmf
)
from .smry import CUBres, CUBsample

def pmfi(m, beta, xi, Y):
    n = Y.shape[0]
    pi_i = logis(Y, beta)
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = cub_pmf(m=m, pi=pi_i[i],
            xi=xi)
    return p

def pmf(m, beta, xi, Y):
    p = pmfi(m, beta, xi, Y)
    pr = p.mean(axis=0)
    return pr

def prob(m, sample, Y, beta, xi):
    p = (
        logis(Y=Y, param=beta)*
        (bitxi(m=m, sample=sample, xi=xi) - 1/m)
        ) + 1/m
    return p

def loglik(m, sample, Y, beta, xi):
    p = probbit(m, xi)
    pn = p[sample-1]
    eta = logis(Y, param=beta)
    l = np.sum(np.log(eta*(pn-1/m)+1/m))
    return l

def draw(m, n, beta, xi, Y, seed=None):
    """
    generate random sample from CUB model
    """
    #np.random.seed(seed)
    assert n == Y.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, beta=beta,
        xi=xi, Y=Y)
    #print("n", n)
    for i in range(n):
        if seed is not None:
            np.random.seed(seed*i)
        rv[i] = np.random.choice(
            choices(m=m),
            size=1,
            replace=True,
            p=theoric_i[i]
        )
    f = freq(m=m, sample=rv)
    theoric = pmf(m=m,beta=beta,xi=xi,Y=Y)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        beta, [xi]
    ))
    par_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["xi"]
    ))
    sample = CUBsample(
        model="CUB(Y0)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, Y=Y, diss=diss,
        theoric=theoric
    )
    return sample

def varcov(m, sample, Y, beta, xi):
    vvi = (m-sample)/xi-(sample-1)/(1-xi)
    ui = (m-sample)/(xi**2)+(sample-1)/((1-xi)**2)
    qi = 1/(m*prob(m=m,sample=sample,Y=Y,beta=beta,xi=xi))
    ei = logis(Y=Y,param=beta)
    qistar = 1-(1-ei)*qi
    eitilde = ei*(1-ei)
    qitilde = qistar*(1-qistar)
    ff = eitilde-qitilde
    g10 = vvi*qitilde
    YY = addones(Y)
    i11 = YY.T @ hadprod(YY,ff) # ALTERNATIVE  YY*ff does not work
    i12 = -YY.T @ g10
    i22 = np.sum(ui*qistar-(vvi**2)*qitilde)
    # Information matrix
    nparam = colsof(YY) + 1
    matinf = np.ndarray(shape=(nparam, nparam))
    matinf[:] = np.nan
    for i in range(nparam-1):
        matinf[i,:] = np.concatenate((i11[i,:],[i12[i]])).T
    matinf[nparam-1,:] = np.concatenate((i12.T,[i22])).T
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        varmat = np.linalg.inv(matinf)
    return varmat

def effe10(beta, esterno10):
    tauno = esterno10[:,0]
    covar = esterno10[:,1:]
    covbet = covar @ beta
    r = np.sum(
        np.log(1+np.exp(-covbet))
        +(1-tauno)*covbet
    )
    return r

def mle(sample, m, Y,
    gen_pars=None,
    maxiter=500,
    tol=1e-4):
    # start datetime
    start = dt.datetime.now()
    # cast sample to numpy array
    sample = np.array(sample)
    # model preference choices
    #R = choices(m)
    # observed absolute frequecies
    f = freq(sample, m)
    # sample size
    n = sample.size
    #aver = np.mean(sample)
    # add a column of 1
    YY = np.c_[np.ones(Y.shape[0]), Y]
    # number of covariates
    p = colsof(Y)
    # init params
    pi, xijj = init_theta(f, m)
    beta0 = np.log(pi/(1-pi))
    betajj = np.concatenate((
        [beta0],
        np.repeat(.1, p)
    ))
    # init loglik
    l = loglik(m=m, sample=sample, Y=Y, beta=betajj, xi=xijj)
    # start E-M algorithm
    niter = 1
    while niter < maxiter:
        lold = l
        bb = probbit(m=m, xi=xijj)
        vettn = bb[sample-1]
        aai = -1 + 1/logis(Y=Y, param=betajj)
        ttau = 1/(1 + aai/(m*vettn))
        averpo = np.sum(sample*ttau)/np.sum(ttau)
        beta = betajj
        esterno10 = np.c_[ttau, YY]
        optimbeta = minimize(
            effe10, x0=beta, args=(esterno10),
            #method="Nelder-Mead"
            #method="BFGS"
        )
        betajj = optimbeta.x
        xijj = (m-averpo)/(m-1)
        l = loglik(m=m, sample=sample, Y=Y, beta=betajj, xi=xijj)
        deltal = abs(lold-l)
        if deltal < tol:
            break
        else:
            lold = l
        niter += 1
    beta = betajj
    xi = xijj
    # variance-covariance matrix
    varmat = varcov(m=m, sample=sample, Y=Y, beta=beta, xi=xi)
    end = dt.datetime.now()
    # standard errors
    stderrs = np.sqrt(np.diag(varmat))
    # Wald statistics
    wald = np.concatenate([beta, [xi]])/stderrs
    # p-value
    pval = 2*(sps.norm().sf(abs(wald)))
    # mean loglikelihood
    muloglik = l/n
    # names for summary
    beta_names = np.concatenate([
        ["constant"],
        Y.columns])
    est_names = np.concatenate((
        beta_names, ["xi"]
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        np.repeat(None, p),
        ["Feeling"]
    ))
    # Akaike Information Criterion
    AIC = aic(l=l, p=p+2)
    # Bayesian Information Criterion
    BIC = bic(l=l, p=p+2, n=n)
    # test
    loglikuni = luni(m=m,n=n)
    #logliksat = lsat(n=n,f=f)
    #dev = 2*(logliksat-l)
    theoric = pmf(m=m, beta=beta, xi=xi, Y=Y)
    diss = dissimilarity(f/n, theoric)
    estimates = np.concatenate((
        beta, [xi]
    ))
    
    res = CUBresCUBY0(
            model="CUB(Y0)",
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
            #logliksat=logliksat,
            # loglikbin=loglikbin,
            # Ebin=Ebin, Ecub=Ecub, Ecub0=Ecub0,
            #dev=dev,
            AIC=AIC, BIC=BIC,
            #ICOMP=ICOMP,
            seconds=(end-start).total_seconds(),
            time_exe=start,
            # rho=rho,
            sample=sample, f=f,
            varmat=varmat,
            diss=diss, Y=Y,
            gen_pars=gen_pars
            # pi_gen=pi_gen, xi_gen=xi_gen
        )
    return res

class CUBresCUBY0(CUBres):
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        title = "MARGINAL PROBABILITY MASS\n"
        title += f"{self.model} model    "
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
    