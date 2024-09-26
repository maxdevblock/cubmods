# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace, invalid-unary-operand-type
"""
CUB models in Python.
Module for CUBSH2 (Combination of Uniform
and Binomial with two Shelter Choices) with covariates.

Description:
============
    This module contains methods and classes
    for CUBSH2_YWXU model family.

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
@Institution: Università degli Studi di Brescia
@Affiliation: Analytics and Data Science for Economics and Management
@Date:        2024
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""

import datetime as dt
import numpy as np
import pandas as pd
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
    addones, 
    #hadprod, 
    aic, bic,
    #lsat, 
    luni, dissimilarity,
    #lsatcov
)
#TODO: change to .cubsh2 when production
from ._cubsh2 import (
    pmf as pmf_cubsh2,
    pdz_to_p123
)
from .smry import CUBres, CUBsample

def pmf(m, sh1, sh2, beta, gamma, omega,
    psi, Y, W, X, U):
    p = pmfi(m, sh1, sh2, beta, gamma, omega,
        psi, Y, W, X, U)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, sh1, sh2, beta, gamma, omega,
    psi, Y, W, X, U):
    pi = logis(Y, beta)
    xi = logis(W, gamma)
    delta = logis(X, omega)
    zeta = logis(U, psi)
    pi1, pi2, pi3 = pdz_to_p123(pi, delta, zeta)
    n = Y.shape[0]
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = pmf_cubsh2(
            m=m, sh1=sh1, sh2=sh2,
            pi1=pi1[i], pi2=pi2[i], pi3=pi3[i],
            xi=xi[i]
        )
    return p

def draw(m, n, sh1, sh2, beta, gamma, omega, psi,
    Y, W, X, U, df, formula, seed=None):
    """
    generate random sample from CUB model
    """
    #np.random.seed(seed)
    #assert n == W.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, sh1=sh1, sh2=sh2,
        beta=beta, gamma=gamma, omega=omega, psi=psi,
        Y=Y, W=W, X=X, U=U)
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
        beta, gamma, omega, psi
    ))
    par_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        X.columns,
        ["constant"],
        U.columns,
    ))
    sample = CUBsample(
        model="CUBSH2(YWXU)",
        df=df, formula=formula,
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, diss=diss,
        theoric=theoric
    )
    return sample

def init_theta(m, sample, p_y, p_x, p_u, W):
    f = freq(m=m, sample=sample)
    pi, _ = inipixi(f=f, m=m)
    beta0 = np.log(pi/(1-pi))
    beta = np.concatenate((
        [beta0], np.repeat(0., p_y)
    ))
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    gamma = init_gamma(sample=sample,
        m=m, W=W)
    omega = np.repeat(.1, p_x+1)
    psi = np.repeat(.1, p_u+1)
    return beta, gamma, omega, psi

def prob(m, sample, sh1, sh2, Y, W, X, U,
    beta, gamma, omega, psi):
    pi = logis(Y, beta)
    delta = logis(X, omega)
    zeta = logis(U, psi)
    pi1, pi2, pi3 = pdz_to_p123(pi, delta, zeta)
    pi4 = 1 - pi1 - pi2 - pi3
    D1 = (sample==sh1).astype(int)
    D2 = (sample==sh2).astype(int)
    bg = bitgamma(sample=sample, m=m, 
        W=W, gamma=gamma)
    p = pi1*bg + pi2/m + pi3*D1 + pi4*D2
    return p

def varcov(sample, m, sh1, sh2, Y, W, X, U,
    beta, gamma, omega, psi):
    # Pr(R=ri|theta,covariates)
    P = prob(m, sample, sh1, sh2, Y, W, X, U,
        beta, gamma, omega, psi)
    if isinstance(P, pd.Series):
        P = P.values
    # c_vartheta
    c_bet = logis(Y, beta)
    c_gam = logis(W, gamma)
    c_ome = logis(X, omega)
    c_psi = logis(U, psi)
    # tildec_vartheta
    tildec_bet = c_bet*(1-c_bet)
    tildec_gam = c_gam*(1-c_gam)
    tildec_ome = c_ome*(1-c_ome)
    tildec_psi = c_psi*(1-c_psi)
    # brevec_vartheta
    brevec_bet = tildec_bet*(1 - 2*c_bet)
    brevec_ome = tildec_ome*(1 - 2*c_ome)
    brevec_psi = tildec_psi*(1 - 2*c_psi)
    # binomial component
    B = bitgamma(sample=sample, m=m,
        W=W, gamma=gamma)
    if isinstance(B, pd.Series):
        B = B.values
    a = (sample-1) - (m-1)*(1-c_gam)
    d = B*((m-1)*tildec_gam - a**2)
    # Shelter Choices
    D1 = (sample==sh1).astype(int)
    D2 = (sample==sh2).astype(int)
    # addones
    YY = addones(Y)
    WW = addones(W)
    XX = addones(X)
    UU = addones(U)
    # f()
    f_bet = (1-c_ome)*tildec_bet*(B-1/m)
    f_gam = (c_ome-1)*c_bet*a*B
    f_psi = c_ome*tildec_psi*(D1-D2)
    f_ome = tildec_ome*(c_psi*(D1-D2)+D2-c_bet*(B-1/m)-1/m)
    # g()
    g_betbet = (1-c_ome)*brevec_bet*(B-1/m)
    g_betome = tildec_ome*tildec_bet*(1/m-B)
    g_betpsi = 0
    g_betgam = (c_ome-1)*tildec_bet*a*B
    g_gamgam = (c_ome-1)*c_bet*d
    g_gampsi = 0
    g_gamome = tildec_ome*c_bet*a*B
    g_psipsi = c_ome*brevec_psi*(D1-D2)
    g_psiome = tildec_ome*tildec_psi*(D1-D2)
    g_omeome = brevec_ome*(c_psi*(D1-D2)+D2-c_bet*(B-1/m)-1/m)
    # h()
    h_betbet = (g_betbet*P - f_bet*f_bet)/(P**2)
    h_betome = (g_betome*P - f_bet*f_ome)/(P**2)
    h_betpsi = (g_betpsi*P - f_bet*f_psi)/(P**2)
    h_betgam = (g_betgam*P - f_bet*f_gam)/(P**2)
    h_gamgam = (g_gamgam*P - f_gam*f_gam)/(P**2)
    h_gampsi = (g_gampsi*P - f_gam*f_psi)/(P**2)
    h_gamome = (g_gamome*P - f_gam*f_ome)/(P**2)
    h_psipsi = (g_psipsi*P - f_psi*f_psi)/(P**2)
    h_psiome = (g_psiome*P - f_psi*f_ome)/(P**2)
    h_omeome = (g_omeome*P - f_ome*f_ome)/(P**2)
    # infmat structure
    # | be0be0 | :: | be0ga0 | :: | be0om0 | :: | be0ps0 | :: |
    # |   ::   |    |   ::   |    |   ::   |    |   ::   |    |
    # | ga0be0 | :: | ga0ga0 | :: | ga0om0 | :: | ga0ps0 | :: |
    # |   ::   |    |   ::   |    |   ::   |    |   ::   |    |
    # | om0be0 | :: | om0ga0 | :: | om0om0 | :: | om0ps0 | :: |
    # |   ::   |    |   ::   |    |   ::   |    |   ::   |    |
    # | ps0be0 | :: | ps0ga0 | :: | ps0om0 | :: | ps0ps0 | :: |
    # |   ::   |    |   ::   |    |   ::   |    |   ::   |    |
    # infmat diagonal
    i11 = YY.T @ (YY * h_betbet.reshape(h_betbet.size,1))
    i22 = WW.T @ (WW * h_gamgam.reshape(h_gamgam.size,1))
    i33 = XX.T @ (XX * h_omeome.reshape(h_omeome.size,1))
    i44 = UU.T @ (UU * h_psipsi.reshape(h_psipsi.size,1))
    # others
    i12 = YY.T @ (WW * h_betgam.reshape(h_betgam.size,1))
    i13 = YY.T @ (XX * h_betome.reshape(h_betome.size,1))
    i14 = YY.T @ (UU * h_betpsi.reshape(h_betpsi.size,1))
    i23 = WW.T @ (XX * h_gamome.reshape(h_gamome.size,1))
    i24 = WW.T @ (UU * h_gampsi.reshape(h_gampsi.size,1))
    i34 = XX.T @ (UU * h_psiome.reshape(h_psiome.size,1))
    i21 = i12.T
    i31 = i13.T
    i41 = i14.T
    i32 = i23.T
    i42 = i24.T
    i43 = i34.T

    matinf = -np.r_[
        np.c_[i11, i12, i13, i14],
        np.c_[i21, i22, i23, i24],
        np.c_[i31, i32, i33, i34],
        np.c_[i41, i42, i43, i44],
    ]
    #print(matinf)
    varmat = np.ndarray(shape=matinf.shape)
    varmat[:] = np.nan
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        #TODO: check the sign of inverted matinf
        varmat = np.linalg.inv(matinf)
    return varmat

def loglik(m, sample, sh1, sh2, Y, W, X, U,
    beta, gamma, omega, psi):
    p = prob(m, sample, sh1, sh2, Y, W, X,
        U, beta, gamma, omega, psi)
    l = np.sum(np.log(p))
    return l

def Q1(param, tau1, tau2, tau3, Y, X, U,
    bb, oo):
    #TODO: check indexes
    beta = param[:(bb+1)]
    omega = param[(bb+1):(bb+1+oo+1)]
    psi = param[(bb+1+oo+1):]
    #print(beta, omega, psi)
    tau4 = 1 - tau1 - tau2 - tau3
    pi = logis(Y, beta)
    delta = logis(X, omega)
    zeta = logis(U, psi)
    pi1, pi2, pi3 = pdz_to_p123(pi, delta, zeta)
    pi4 = 1 - pi1 - pi2 - pi3
    esse1 = (tau1*np.log(pi1)).sum()
    esse2 = (tau2*np.log(pi2)).sum()
    esse3 = (tau3*np.log(pi3)).sum()
    esse4 = (tau4*np.log(pi4)).sum()
    esse = -(esse1+esse2+esse3+esse4)
    return esse

def Q2(param, dati2, m):
    tau2 = dati2[0]
    sample = dati2[1]
    W = dati2[2]
    bg = bitgamma(sample=sample, m=m, 
        W=W, gamma=param)
    return -(tau2*np.log(bg)).sum()

def mle(m, sample, sh1, sh2, Y, W, X, U,
    df, formula,
    ass_pars=None,
    maxiter=1000, tol=1e-4):
    start = dt.datetime.now()
    n = sample.size
    # rank = pd.Series(sample).rank(method="dense")
    # rank = rank.astype(int).values
    py = colsof(Y)
    pw = colsof(W)
    px = colsof(X)
    pu = colsof(U)
    beta, gamma, omega, psi = init_theta(m=m, sample=sample, p_y=py, p_x=px, p_u=pu, W=W)
    betomepsi = np.concatenate((beta, omega, psi))
    #print(betomepsi)
    l = loglik(m, sample, sh1, sh2, Y, W, X, U,
               beta, gamma, omega, psi)
    
    niter = 1
    while niter < maxiter:
        lold = l
        pi = logis(Y, beta)
        delta = logis(X, omega)
        zeta = logis(U, psi)

        pi1, pi2, pi3 = pdz_to_p123(pi, delta, zeta)
        pi4 = 1 - pi1 - pi2 - pi3
        
        P1 = bitgamma(sample=sample, m=m, 
            W=W, gamma=gamma)
        P2 = 1/m
        P3 = (sample==sh1).astype(int)
        P4 = (sample==sh2).astype(int)
        
        tau1 = pi1*P1
        tau2 = pi2*P2
        tau3 = pi3*P3
        tau4 = pi4*P4
        den = tau1+tau2+tau3+tau4
        
        tau1 /= den
        tau2 /= den
        tau3 /= den
        tau4 /= den
        
        dati1 = (tau1,tau2,tau3,Y,X,U,py,px)
        dati2 = [tau1,sample,W]
        
        optim1 = minimize(
            Q1, x0=betomepsi,
            args=dati1
        )
        optim2 = minimize(
            Q2, x0=gamma,
            args=(dati2, m)
        )
        
        betomepsi = optim1.x
        beta = betomepsi[:beta.size]
        omega = betomepsi[beta.size:beta.size+omega.size]
        psi = betomepsi[-psi.size:]
        gamma = optim2.x
        
        l = loglik(m, sample, sh1, sh2, Y, W, X, U,
               beta, gamma, omega, psi)
        testl = abs(l-lold)
        if testl <= tol:
            break
        niter += 1
    muloglik = l/n
    varmat = varcov(sample=sample, m=m,
        sh1=sh1, sh2=sh2, Y=Y, W=W, X=X, U=U,
        beta=beta, gamma=gamma, omega=omega, psi=psi)
    stderrs = np.sqrt(np.diag(varmat))
    estimates = np.concatenate((
        beta, gamma, omega, psi
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
    theoric = pmf(m, sh1, sh2, beta, gamma, omega, psi, Y, W, X, U)
    diss = dissimilarity(f/n, theoric)
    est_names = np.concatenate((
        ["constant"],
        Y.columns,
        ["constant"],
        W.columns,
        ["constant"],
        X.columns,
        ["constant"],
        U.columns
    ))
    e_types = np.concatenate((
        ["Uncertainty (beta)"],
        [None for _ in range(py)],
        ["Feeling (gamma)"],
        [None for _ in range(pw)],
        ["Shelter effect (omega)"],
        [None for _ in range(px)],
        ["1st shelter choice (psi)"],
        [None for _ in range(pu)]
    ))
    end = dt.datetime.now()
    
    return CUBresCUBSH2YWXU(
        model="CUBSH2(YWXU)",
        df=df, formula=formula,
        m=m, n=n, sh=[sh1,sh2], sample=sample,
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
        time_exe=start, ass_pars=ass_pars
    )
    
class CUBresCUBSH2YWXU(CUBres):
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        else:
            fig = None
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
        #     title += fr"Assumed($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
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
        # if self.ass_pars is not None:
        #     pi_gen = self.ass_pars["pi"]
        #     gamma_gen = self.ass_pars["gamma"]
        #     phi_gen = self.ass_pars["phi"]
        #     p_gen = pmf(m=self.m, pi=pi_gen,
        #         gamma=gamma_gen, phi=phi_gen,
        #         W=self.W)
        #     ax.stem(R, p_gen, linefmt="--r",
        #     markerfmt="none", label="assumed")

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        if fig is not None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
    
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