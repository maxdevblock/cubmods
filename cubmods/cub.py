# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial).

Description
============
    This module contains methods and classes
    for CUB model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

    :math:`\Pr(R=r|\pmb{\theta}) = \pi \dbinom{m-1}{r-1}(1-\xi)^{r-1}\xi^{m-r}+\dfrac{1-\pi}{m}`

Manual and Examples
==========================
  - Manual https://github.com/maxdevblock/cubmods/blob/main/Manual/02_cub_family.md


References
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

:Author:      Massimo Pierini
:Institution: Universitas Mercatorum
:Affiliation: Graduand in Statistics & Big Data (L41)
:Date:        2023-24
:Credit:      Domenico Piccolo, Rosaria Simone
:Contacts:    cub@maxpierini.it

Classes and Functions
==============
"""

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
    ParameterOutOfBoundsError,
    InvalidSampleSizeError,
    lsat, luni, aic, bic,
    #chisquared,
)
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def pmf(m, pi, xi):
    r"""Probability distribution of a specified CUB model.

    :math:`\Pr(R = r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the vector of the probability distribution of a CUB model.
    :rtype: numpy array
    """
    R = choices(m)
    #print(m, pi, xi, R)
    p = pi*binom(m-1, R-1) * (1-xi)**(R-1) * xi**(m-R) + (1-pi)/m
    return p

def proba(m, pi, xi, r):
    r"""Probability :math:`\Pr(R = r | \pmb\theta)` of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param r: ordinal value (must be :math:`1 \leq r \leq m`)
    :type r: int
    :return: the probability :math:`\Pr(R = r | \pmb\theta)`
    :rtype: float
    """
    #print(m, pi, xi, R)
    p = pi*binom(m-1, r-1) * (1-xi)**(r-1) * xi**(m-r) + (1-pi)/m
    #print(p)
    return p

def cmf(m, pi, xi):
    r"""Cumulative probability of a specified CUB model.

    :math:`\Pr(R \geq r | \pmb\theta),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: an array of the CMF for the specified model
    :rtype: numpy array
    """
    return pmf(m, pi, xi).cumsum()

def mean(m, pi, xi):
    r"""Expected value of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the expected value of the model
    :rtype: float
    """
    return (m+1)/2 + pi*(m-1)*(1/2-xi)

def var(m, pi, xi):
    r"""Variance of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the variance of the model
    :rtype: float
    """
    v =  (m-1)*(pi*xi*(1-xi) + (1-pi)*((m+1)/12+pi*(m-1)*(xi-1/2)**2))
    return v

def std(m, pi, xi):
    r"""Standard deviation of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the standard deviation of the model
    :rtype: float
    """
    return np.sqrt(var(m, pi, xi))

def skew(pi, xi):
    r"""Skewness normalized :math:`\eta` index

    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the skewness of the model
    :rtype: float
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
    r"""The median of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the median of the model
    :rtype: float
    """
    R = choices(m)
    cp = cmf(m, pi, xi)
    M = R[cp>.5][0]
    if M > R.max():
        M = R.max()
    return M

def gini(m, pi, xi):
    r"""The Gini index of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the Gini index of the model
    :rtype: float
    """
    ssum = 0
    for r in choices(m):
        ssum += proba(m, pi, xi, r)**2
    return m*(1-ssum)/(m-1)

def laakso(m, pi, xi):
    r"""The Laakso index of a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the Laakso index of the model
    :rtype: float
    """
    g = gini(m, pi, xi)
    return g/(m - (m-1)*g)

def _rvs(m, pi, xi, n):
    r"""
    :DEPRECATED:
    """
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=pmf(m, pi, xi)
        )
    return rv

def loglik(m, pi, xi, f):
    r"""Compute the log-likelihood function of a CUB model without 
    covariates for a given absolute frequency distribution.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param f: array of absolute frequency distribution
    :type f: array of int
    :return: the log-likelihood value
    :rtype: float
    """
    L = pmf(m, pi, xi)
    l = (f*np.log(L)).sum()
    return l

def varcov(m, pi, xi, ordinal):
    r"""Compute the variance-covariance matrix of parameter 
    estimates of a CUB model without covariates.

    :references:

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param ordinal: array of ordinal responses
    :type ordinal: array of int
    :return: the variance-covariance matrix of the CUB model
    :rtype: numpy ndarray
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
    #TODO: create matrix from array in R style?
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
    r"""Preliminary estimators for CUB models without covariates.

    Computes preliminary parameter estimates of a CUB model without covariates for given ordinal
    responses. These preliminary estimators are used within the package code to start the E-M algorithm.

    :param f: array of the absolute frequencies of given ordinal responses
    :type f: array of int
    :param m: number of ordinal categories
    :type m: int
    :return: a tuple of :math:`(\pi^{(0)}, \xi^{(0)})`
    """
    #pi = .5
    #xi = (m-avg)/(m-1)
    F = f/f.sum()
    xi = 1 + (.5 - (np.argmax(F)+1))/m
    ppp = probbit(m, xi)
    pi = np.sqrt( (np.sum(F**2)-1/m) /
        (np.sum(ppp**2)-1/m) )
    pi = min([pi, .99])
    return pi, xi

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, pi, xi, n, seed=None):
    r"""Draw a random sample from a specified CUB model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param n: number of ordinal responses to be drawn
    :type n: int
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    if m<= 3:
        print("ERR: Number of ordered categories should be at least 4")
        raise InvalidCategoriesError(m=m, model="cub")
    if xi < 0 or xi > 1:
        raise ParameterOutOfBoundsError("xi", xi)
    if pi < 0 or pi > 1:
        raise ParameterOutOfBoundsError("pi", pi)
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
    tol=1e-4,):
    r"""Main function for CUB models without covariates.

    Function to estimate and validate a CUB model without covariates for given ordinal responses.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param gen_pars: dictionary of hypothesized parameters, defaults to None
    :type gen_pars: dictionary, optional
    :param maxiter: maximum number of iterations allowed for running the optimization algorithm
    :type maxiter: int
    :param tol: fixed error tolerance for final estimates
    :type tol: float
    :return: an instance of ``CUBresCUB00`` (see the Class for details)
    :rtype: object
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
    # model ordinal categories
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
    AIC = aic(l=l, p=2)
    # Bayesian Information Criterion
    BIC = bic(l=l, p=2, n=n)
    # mean loglikelihood
    muloglik = l/n
    # loglik of null model (uniform)
    loglikuni = luni(m=m, n=n)
    # loglik of saturated model
    logliksat = lsat(f=f, n=n)
    # loglik of shiftet binomial
    #xibin = (m-sample.mean())/(m-1)
    #TODO: compute loglikbin too?
    #loglikbin = loglik(m, 1, xibin, f)
    # Explicative powers
    #Ebin = (loglikbin-loglikuni)/(logliksat-loglikuni)
    #Ecub = (l-loglikbin)/(logliksat-loglikuni)
    #Ecub0 = (l-loglikuni)/(logliksat-loglikuni)
    # deviance from saturated model
    dev = 2*(logliksat-l)
    # ICOMP metrics
    #npars = 2
    #trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + npars*np.log(trvarmat/npars) - np.log(np.linalg.det(varmat))
    # coefficient of correlation
    rho = varmat[0,1]/np.sqrt(varmat[0,0]*varmat[1,1])
    theoric = pmf(m=m, pi=pi, xi=xi)
    #TODO: add dissimilarity from generating model
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
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """

    def plot_ordinal(self, figsize=(7, 5), kind="bar",
        ax=None,saveas=None):
        """Plots relative frequencies of observed sample, estimated probability distribution and,
        if provided, probability distribution of a known model.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param kind: choose a barplot (``'bar'`` default) of a scatterplot (``'scatter'``)
        :type kind: str
        :param ax: matplotlib axis, if None a new figure will be created, defaults to None
        :type ax: matplolib ax, optional
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
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

    def plot_confell(self, figsize=(7, 5),
        ci=.95, equal=True,
        magnified=False, ax=None,
        saveas=None):
        r"""Plots the estimated parameter values in the parameter space and
        the asymptotic confidence ellipse.
        
        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipse
        :type ci: float
        :param equal: if the plot must have equal aspect
        :type equal: bool
        :param magnified: if False the limits will be the entire parameter space, otherwise let matplotlib choose the limits
        :type magnified: bool
        :param ax: matplotlib axis, if None a new figure will be created, defaults to None
        :type ax: matplolib ax, optional
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )

        if equal:
            ax.set_aspect("equal")
        ax.set_title(fr"Corr($\pi,\xi$)= {self.rho:.4f}")

        pi = self.estimates[0]
        xi = self.estimates[1]

        ax.set_xlabel(r"$(1-\pi)$  uncertainty")
        ax.set_ylabel(r"$(1-\xi)$  feeling")

        ax.plot(1-pi, 1-xi,
            ".b",ms=20, alpha=.5,
            label="estimated")
        if self.gen_pars is not None:
            ax.scatter(1-self.gen_pars['pi'], 1-self.gen_pars['xi'],
                facecolor="None",
                edgecolor="r", s=200, label="generating")

        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(2)
            # increase tick width
            ax.tick_params(width=2)

        #alpha = 1 - ci
        #z = abs(sps.norm().ppf(alpha/2))
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
        r"""Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param ci: level :math:`(1-\alpha/2)` for the confidence ellipse
        :type ci: float
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
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

    def _save(self, fname):
        """
        :DEPRECATED:
        """
        filename = f"{fname}.cub.fit"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Fitting saved to {filename}")
