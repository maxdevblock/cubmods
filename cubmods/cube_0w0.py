# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
.. _cube0w0-module:

CUB models in Python.
Module for CUBE (Combination of Uniform
and Beta-Binomial) with covariates for the feeling component.

Description:
============
    This module contains methods and classes
    for CUBE_0W0 model family.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

    :math:`\Pr(R_i=r|\pmb{\theta};\pmb w_i) = \pi \beta e(\xi_i,\phi)+\dfrac{1-\pi}{m},\; r=1,2,\ldots,m`

    :math:`\xi_i = \dfrac{1}{1+\exp\{-\pmb w_i \pmb \gamma\}}`

Manual and Examples
==========================
    - `Models manual <manual.html#cube-with-covariates>`__


References:
===========
    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        iannario2014modelling
        iannario2015detecting
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

  
List of TODOs:
==============
  - Manual and Examples
  - Remove unused imports

Credits
==============
    :Author:      Massimo Pierini

    :Date:        2023-24
    :Credits:     Domenico Piccolo, Rosaria Simone
    :Contacts:    cub@maxpierini.it

Classes and Functions
=====================
"""

import datetime as dt
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
from statsmodels.tools.numdiff import approx_hess
import matplotlib.pyplot as plt
from .general import (
    logis, dissimilarity,
    aic, bic, luni, 
    #lsat,
    freq, choices, 
    #lsatcov,
    #addones, colsof,
)
from .cube import (
    betar,
    #init_theta as ini_cube,
    mle as mle_cube
)
from .cub_0w import init_gamma
from .smry import CUBres, CUBsample

def pmfi(m, pi, gamma, phi, W):
    r"""Probability distribution for each subject of a specified CUBE model 
    with covariates for feeling only.
    
    Auxiliary function of ``.draw()``.

    :math:`\Pr(R_i=r|\pmb\theta; \pmb T_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    n = W.shape[0]
    xi = logis(W, gamma)
    p = np.ndarray(shape=(n, m))
    for i in range(n):
        pBe = betar(m=m, xi=xi[i], phi=phi)
        p[i,:] = pi*(pBe-1/m) + 1/m
    return p

def pmf(m, pi, gamma, phi, W):
    r"""Average probability distribution of a specified CUB model 
    with covariates for the feeling component.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R_i=r|\pmb\theta; \pmb T_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :return: the array of the average probability distribution
    :rtype: numpy array
    """
    p = pmfi(m, pi, gamma, phi, W).mean(
        axis=0)
    #print(p_i)
    return p

def betabinomialxi(m, sample, xivett, phi):
    r"""
    Beta-Binomial probabilities of ordinal responses, given feeling parameter for each observation.

    Compute the Beta-Binomial probabilities of given ordinal responses, with feeling 
    parameter specified for each observation, 
    and with the same overdispersion parameter for all the responses.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses. Missing values are not allowed: they should be preliminarily deleted
    :type sample: array
    :param xivett: array of feeling parameters of the Beta-Binomial distribution for given ordinal responses
    :type xivett: array
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :return: array of the same length as ordinal: each entry is the Beta-Binomial probability for the given observation 
        for the corresponding feeling and overdispersion parameters.
    :rtype: array
    """
    n = sample.size
    betabin = np.repeat(np.nan, n)
    for i in range(n):
        bebeta = betar(m=m, xi=xivett[i],
        phi=phi)
        betabin[i] = bebeta[sample[i]-1]
    return np.array(betabin)

def draw(m, pi, gamma, phi, W,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUBE model.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param n: number of ordinal responses to be drawn
    :type n: int
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    #np.random.seed(seed)
    assert len(gamma) == W.shape[1]+1
    n = W.shape[0]
    if seed == 0:
        print("Seed cannot be zero. "
        "Modified to 1.")
        seed = 1
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, pi=pi,
        gamma=gamma, phi=phi, W=W)
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
    theoric = pmf(m=m, pi=pi,
        gamma=gamma, phi=phi, W=W)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        [pi], gamma, [phi]
    ))
    par_names = np.concatenate((
        ["pi"],
        ["constant"],
        W.columns,
        ["phi"]
    ))
    p_types = np.concatenate((
        ["Uncertainty"],
        np.repeat(["Feeling"], len(gamma)),
        ["Overdispesion"]
    ))
    sample = CUBsample(
        model="CUBE(0W0)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        p_types=p_types,
        seed=seed, diss=diss,
        theoric=theoric, df=df,
        formula=formula
    )
    return sample

def prob(m, sample, W, pi, gamma, phi):
    r"""Probability distribution of a CUBE model with covariates for feeling.

    Compute the probability distribution of a CUB model with covariates for both the feeling 
    and the uncertainty components. Auxiliary function of ``.loglik()``

    :math:`\Pr(R_i=r_i|\pmb\theta;\pmb T_i),\; i=1 \ldots n`

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param sample: array of ordinal responses
    :type sample: array of int
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    xivett = logis(Y=W, param=gamma)
    p = pi*(betabinomialxi(m=m, sample=sample,
        xivett=xivett, phi=phi)-1/m)+1/m
    return p

def loglik(m, sample, W, pi, gamma, phi):
    r"""Log-likelihood function of CUBE model with covariates only for feeling.

    Compute the log-likelihood function of a CUBE model for ordinal data with subjects' 
    covariates only for feeling.

    :param m: number of ordinal categories
    :type m: int
    :param pi: uncertainty parameter :math:`\pi`
    :type pi: float
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :param phi: overdispersion parameter :math:`\phi`
    :type phi: float
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param sample: array of ordinal responses
    :type sample: array of int
    :return: the log-likelihood value
    :rtype: float
    """
    p = prob(m=m, sample=sample, W=W,
        pi=pi, gamma=gamma, phi=phi)
    l = np.sum(np.log(p))
    return l

def init_theta(m, sample, W, maxiter, tol):
    r"""Preliminary estimates of parameters for CUBE models with covariates only for feeling.

    Compute preliminary parameter estimates of a CUBE model with covariates only for feeling, given
    ordinal responses. These estimates are set as initial values to start the corresponding E-M algorithm within the package.
    Preliminary estimates for the uncertainty and the overdispersion parameters are computed by short runs of EM. 
    As to the feeling component, it considers the nested CUB model with covariates and calls \code{\link{inibestgama}} to derive initial estimates for the coefficients
    of the selected covariates for feeling.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array of int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param maxiter: maximum number of iterations allowed for preliminary iterations
    :type maxiter: int
    :param tol: fixed error tolerance for final estimates for preliminary iterations
    :type tol: float
    :return: a tuple of :math:`(\pi^{(0)}, \pmb \gamma^{(0)}, \phi^{(0)})`, where :math:`\pi^{(0)}` is the initial 
        estimate for the uncertainty parameter, 
        :math:`\pmb \gamma^{(0)}` is the vector of initial estimates for the feeling component (including an intercept 
        term in the first entry),
        and :math:`\phi^{(0)}` is the initial estimate for the overdispersion parameter.
    "rtype": tuple
    """
    gamma = init_gamma(m=m, sample=sample,
        W=W)
    res_cube = mle_cube(m=m, sample=sample,
        maxiter=maxiter, tol=tol,
        df=None, formula=None)
    pi = res_cube.estimates[0]
    phi = res_cube.estimates[2]
    return pi, gamma, phi

def effe(pars, sample, W, m):
    r"""Auxiliary function for the log-likelihood estimation of CUBE models with covariates 
    only for the feeling component.

    Compute the opposite of the scalar function that is maximized when running the 
    E-M algorithm for CUBE models with covariates only for the feeling component.

    :param pars: array of length equal to ``W.index.size+3`` whose entries are the initial parameters estimates
    :type pars: array
    :param sample: array of ordinal responses
    :type sample: array of int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param m: number of ordinal categories
    :type m: int
    :return: negative log-likelihood
    :rtype: float
    """
    pi = pars[0]
    gamma = pars[1:-1]
    phi = pars[-1]
    l = loglik(m, sample, W, pi, gamma, phi)
    return -l

def mle(sample, m, W, df, formula,
    ass_pars=None,
    maxiter=1000, tol=1e-6):
    r"""Main function for CUBE models with covariates only for feeling

    Estimate and validate a CUBE model for ordinal data, with covariates only for explaining the
    feeling component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param W: dataframe of covariates for explaining the feeling component
    :type W: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param ass_pars: dictionary of hypothesized parameters, defaults to None
    :type ass_pars: dictionary, optional
    :param maxiter: maximum number of iterations allowed for preliminary iterations
    :type maxiter: int
    :param tol: fixed error tolerance for final estimates for preliminary iterations;
        the informatio matrix (to compute the variance-covariance matrix) is approximated with ``approx_hess()``
        (see ``statsmodels.tools.numdiff`` for details)
    :type tol: float
    :return: an instance of ``CUBresCUBE0W0`` (see the Class for details)
    :rtype: object
    """
    start = dt.datetime.now()
    n = sample.size
    pi, gamma, phi = init_theta(
        m=m, sample=sample, W=W,
        maxiter=maxiter, tol=tol
    )
    l = loglik(m, sample, W, pi, gamma, phi)
    pars0 = np.concatenate((
        [pi], gamma, [phi]
    ))
    #print(pars0)
    q = gamma.size - 1
    bounds = [(.01, .99)]
    for _ in range(q+1):
        bounds.append((None, None))
    bounds.append((.01, .3))
    optim = minimize(effe, x0=pars0,
        args=(sample, W, m),
        method="L-BFGS-B",
        bounds=bounds
        )
    pars = optim.x
    pi = pars[0]
    gamma = pars[1:-1]
    phi = pars[-1]
    
    infmat = approx_hess(pars, effe,
        args=(sample, W, m))
    varmat = np.ndarray(shape=(pars.size,pars.size))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    
    stderrs = np.sqrt(np.diag(varmat))
    l = loglik(m, sample, W, pi, gamma, phi)
    theoric = pmf(m=m, pi=pi, gamma=gamma,
        phi=phi, W=W)
    f = freq(sample=sample, m=m)
    diss = dissimilarity(f/n, theoric)
    loglikuni = luni(m=m, n=n)
    # logliksat = lsat(n=n, f=f)
    # logliksatcov = lsatcov(
    #     sample=sample,
    #     covars=[W]
    # )
    muloglik = l/n
    # dev = 2*(logliksat-l)
    
    estimates = pars
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    est_names = np.concatenate((
        ["pi"],
        np.concatenate((
            ["constant"],
            [x for x in W.columns]
        )),
        ["phi"]
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        ["Feeling"],
        np.repeat(None, q),
        ["Overdisperson"]
    ))
    AIC = aic(p=estimates.size, l=l)
    BIC = bic(l=l, p=estimates.size, n=n)
    end = dt.datetime.now()
    
    return CUBresCUBE0W0(
        model="CUBE(0W0)",
        n=n, m=m, sample=sample, f=f,
        estimates=estimates,
        stderrs=stderrs,
        wald=wald, pval=pval,
        est_names=est_names,
        e_types=e_types,
        theoric=theoric,
        AIC=AIC, BIC=BIC,
        loglike=l,
        # logliksat=logliksat,
        # logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        muloglik=muloglik,
        diss=diss,
        # dev=dev,
        varmat=varmat,
        seconds=(end-start).total_seconds(),
        time_exe=start,
        ass_pars=ass_pars,
        df=df, formula=formula
    )

class CUBresCUBE0W0(CUBres):
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        """Plots relative average frequencies of observed sample, estimated average probability distribution and,
        if provided, average probability distribution of a known model.

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
        if self.ass_pars is not None: #TODO: implement ass_pars
            pass
            # pi_gen = self.ass_pars["pi"]
            # gamma_gen = self.ass_pars["gamma"]
            # phi_gen = self.ass_pars["phi"]
            # p_gen = pmf(m=self.m, pi=pi_gen,
            #     gamma=gamma_gen, phi=phi_gen,
            #     W=self.W)
            # ax.stem(R, p_gen, linefmt="--r",
            # markerfmt="none", label="assumed")

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
        """Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
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
    