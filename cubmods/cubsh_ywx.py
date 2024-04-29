# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace, invalid-unary-operand-type
"""
CUB models in Python.
Module for CUBSH (Combination of Uniform
and Binomial with Shelter Effect) with covariates.

Description:
============
    This module contains methods and classes
    for CUBSH_YWX model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    TODO: add example


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
#import pandas as pd
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from .cub import (
    init_theta as inipixi
)
from .cub_0w import (
    init_gamma, bitgamma
)
from .general import (
    choices, freq, logis, colsof,
    addones, hadprod, aic, bic,
    #lsat, 
    luni, dissimilarity,
    #lsatcov
)
from .cubsh import (
    pmf as pmf_cubsh,
    pidelta_to_pi1pi2
)
from .smry import CUBres, CUBsample

def pmf(m, sh, beta, gamma, omega,
    Y, W, X):
    p = pmfi(m, sh, beta, gamma, omega,
        Y, W, X)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, sh, beta, gamma, omega,
    Y, W, X):
    pi = logis(Y, beta)
    xi = logis(W, gamma)
    delta = logis(X, omega)
    pi1, pi2 = pidelta_to_pi1pi2(pi, delta)
    n = Y.shape[0]
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = pmf_cubsh(
            m=m, sh=sh,
            pi1=pi1[i], pi2=pi2[i],
            xi=xi[i]
        )
    return p

def draw(m, n, sh, beta, gamma, omega,
    Y, W, X, seed=None):
    """
    generate random sample from CUB model
    """
    #np.random.seed(seed)
    assert n == W.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, sh=sh, beta=beta,
        gamma=gamma, omega=omega,
        Y=Y, W=W, X=X)
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
    theoric = theoric_i.mean(axis=0)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        beta, gamma, omega
    ))
    par_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        X.columns,
    ))
    sample = CUBsample(
        model="CUBSH(YWX)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, W=W, diss=diss,
        theoric=theoric
    )
    return sample

def init_theta(m, sample, p, s, W):
    f = freq(m=m, sample=sample)
    pi, _ = inipixi(f=f, m=m)
    beta0 = np.log(pi/(1-pi))
    beta = np.concatenate((
        [beta0], np.repeat(0., p)
    ))
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    gamma = init_gamma(sample=sample,
        m=m, W=W)
    omega = np.repeat(.1, s+1)
    return beta, gamma, omega

def prob(m, sample, sh, Y, W, X,
    beta, gamma, omega):
    alpha1 = logis(X, omega)
    alpha2 = (1-alpha1)*logis(Y, beta)
    D = (sample==sh).astype(int)
    bg = bitgamma(sample=sample, m=m, 
        W=W, gamma=gamma)
    d = 1 - alpha1 - alpha2
    p = alpha1*D + alpha2*bg + d/m
    return p

def varcov(sample, m, sh, Y, W, X,
    beta, gamma, omega):
    probi = prob(m, sample, sh, Y, W, X,
        beta, gamma, omega)
    vvi = 1/probi
    D = (sample==sh).astype(int)
    pii = logis(Y, beta)
    xii = logis(W, gamma)
    deltai = logis(X, omega)
    bri = bitgamma(sample=sample, m=m,
        W=W, gamma=gamma)
    YY = addones(Y)
    WW = addones(W)
    XX = addones(X)
    npar = colsof(YY)+colsof(WW)+colsof(XX)
    mi = m-sample-(m-1)*xii
    vAA = pii*(1-pii)*(1-deltai)*(bri-1/m)*vvi
    AA = hadprod(YY, vAA)
    vBB = pii*(1-deltai)*mi*bri*vvi
    BB = hadprod(WW, vBB)
    vCC = deltai*(D-probi)*vvi
    CC = hadprod(XX, vCC)
    
    di = pii*(1-pii)*(1-2*pii)*(1-deltai)*(bri-1/m)*vvi
    gi = pii*(1-deltai)*bri*(mi**2-(m-1)*xii)*(1-xii)*vvi
    li = deltai*(1-2*deltai)*(D-probi)*vvi
    ei = pii*(1-pii)*(1-deltai)*bri*mi*vvi
    fi = (-pii)*(1-pii)*deltai*(1-deltai)*(bri-1/m)*vvi
    hi = (-pii)*deltai*(1-deltai)*bri*mi*vvi
    
    i11 = (AA.T @ AA) - (YY.T @ hadprod(YY,di))
    i22 = (BB.T @ BB) - (WW.T @ hadprod(WW,gi))
    i33 = (CC.T @ CC) - (XX.T @ hadprod(XX,li))
    i21 = (BB.T @ AA) - (WW.T @ hadprod(YY,ei))
    i31 = (CC.T @ AA) - (XX.T @ hadprod(YY,fi))
    i32 = (CC.T @ BB) - (XX.T @ hadprod(WW,hi))
    i12 = i21.T
    i13 = i31.T
    i23 = i32.T
    matinf = np.r_[
        np.c_[i11, i12, i13],
        np.c_[i21, i22, i23],
        np.c_[i31, i32, i33]
    ]
    #print(matinf)
    varmat = np.ndarray(shape=(npar, npar))
    varmat[:] = np.nan
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        varmat = np.linalg.inv(matinf)
    return varmat

def loglik(m, sample, sh, Y, W, X,
    beta, gamma, omega):
    p = prob(m, sample, sh, Y, W, X,
        beta, gamma, omega)
    l = np.sum(np.log(p))
    return l

def Q1(param, dati1, p):
    omega = param[:-(p+1)]
    beta = param[-(p+1):]
    tau1 = dati1[0]
    tau2 = dati1[1]
    tau3 = 1 - tau1 - tau2
    X = dati1[2]
    Y = dati1[3]
    alpha1 = logis(X, omega)
    alpha2 = (1-alpha1)*logis(Y, beta)
    alpha3 = 1 - alpha1 - alpha2
    esse1 = (tau1*np.log(alpha1)).sum()
    esse2 = (tau2*np.log(alpha2)).sum()
    esse3 = (tau3*np.log(alpha3)).sum()
    esse = -(esse1+esse2+esse3)
    return esse

def Q2(param, dati2, m):
    tau2 = dati2[0]
    sample = dati2[1]
    W = dati2[2]
    bg = bitgamma(sample=sample, m=m, 
        W=W, gamma=param)
    return -(tau2*np.log(bg)).sum()

def mle(m, sample, sh, Y, W, X,
    gen_pars=None,
    maxiter=500, tol=1e-4):
    start = dt.datetime.now()
    n = sample.size
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    p = colsof(Y)
    q = colsof(W)
    s = colsof(X)
    beta, gamma, omega = init_theta(
        m, sample, p, s, W)
    psi = np.concatenate((omega, beta))
    #print(omega.size, p, s, psi.shape)
    l = loglik(m, sample, sh, Y, W, X,
        beta, gamma, omega)
    
    niter = 1
    while niter < maxiter:
        lold = l
        alpha1 = logis(X, omega)
        alpha2 = (1-alpha1)*logis(Y, beta)
        alpha3 = 1 - alpha1 - alpha2
        
        p1 = (sample==sh).astype(int)
        p2 = bitgamma(sample=sample, m=m, 
            W=W, gamma=gamma)
        p3 = 1/m
        
        num1 = alpha1*p1
        num2 = alpha2*p2
        num3 = alpha3*p3
        den = num1+num2+num3
        
        ttau1 = num1/den
        ttau2 = num2/den
        #ttau3 = 1 - ttau1 - ttau2
        
        dati1 = [ttau1,ttau2,X,Y]
        dati2 = [ttau2,sample,W]
        
        optim1 = minimize(
            Q1, x0=psi,
            args=(dati1, p)
        )
        optim2 = minimize(
            Q2, x0=gamma,
            args=(dati2, m)
        )
        
        psi = optim1.x
        omega = psi[:-beta.size]
        beta = psi[-beta.size:]
        gamma = optim2.x
        
        l = loglik(m, sample, sh, Y, W, X,
            beta, gamma, omega)
        testl = abs(l-lold)
        if testl <= tol:
            break
        niter += 1
    muloglik = l/n
    varmat = varcov(sample=sample, m=m,
        sh=sh, Y=Y, W=W, X=X, beta=beta,
        gamma=gamma, omega=omega)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.concatenate((
        beta, gamma, omega
    ))
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    AIC = aic(l=l, p=wald.size)
    BIC = bic(l=l, p=wald.size, n=n)
    loglikuni = luni(m=m, n=n)
    f = freq(sample=sample, m=m)
    #logliksat = lsat(n=n, f=f)
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[Y,W,X]
    #)
    #dev = 2*(logliksat-l)
    theoric = pmf(m, sh, beta, gamma, omega,
        Y, W, X)
    diss = dissimilarity(f/n, theoric)
    est_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        X.columns
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        [None for _ in range(p)],
        ["Feeling"],
        [None for _ in range(q)],
        ["Shelter effect"],
        [None for _ in range(s)]
    ))
    end = dt.datetime.now()
    
    return CUBresCUBSHYWX(
        model="CUBSH(YWX)",
        m=m, n=n, sh=sh, sample=sample,
        f=f, theoric=theoric,
        niter=niter, maxiter=maxiter,
        tol=tol, stderrs=stderrs,
        est_names=est_names,
        e_types=e_types,
        estimates=estimates,
        wald=wald, pval=pval,
        loglike=l, muloglik=muloglik,
        #logliksat=logliksat,
        #logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        diss=diss, varmat=varmat,
        #dev=dev,
        AIC=AIC, BIC=BIC,
        seconds=(end-start).total_seconds(),
        time_exe=start, gen_pars=gen_pars
    )
    
class CUBresCUBSHYWX(CUBres):
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        #pi = self.estimates[0]
        #xi = self.estimates[1]
        #phi = self.estimates[2]
        title = "AVERAGE ESTIMATED PROBABILITY\n"
        title += f"{self.model} model    "
        title += f"$n={self.n}$\n"
        #title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$ , $\phi={phi:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.3f}"
        #TODO: add dissimilarity from generating model
        # if self.diss_gen is not None:
        #     title += "\n"
        #     title += fr"Gener($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
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
        # if self.gen_pars is not None:
        #     pi_gen = self.gen_pars["pi"]
        #     gamma_gen = self.gen_pars["gamma"]
        #     phi_gen = self.gen_pars["phi"]
        #     p_gen = pmf(m=self.m, pi=pi_gen,
        #         gamma=gamma_gen, phi=phi_gen,
        #         W=self.W)
        #     ax.stem(R, p_gen, linefmt="--r",
        #     markerfmt="none", label="generating")

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
        #self.plot_confell(ci=ci, ax=ax[1])
        #self.plot_confell(
        #    ci=ci, ax=ax[2],
        #    magnified=True, equal=False)
        #plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax