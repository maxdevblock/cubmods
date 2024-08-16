# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
r"""
CUB models in Python.
Module for CUB (Combination of Uniform
and Binomial) with covariates for the uncertainty component.

Description:
============
    This module contains methods and classes
    for CUB_Y0 model family.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

    :math:`\Pr(R=r_i|\pmb{\theta}_i ; \pmb y_i) = \pi_i \dbinom{m-1}{r_i-1}(1-\xi)^{r_i-1}\xi^{m-r_i}+\dfrac{1-\pi_i}{m}`

    :math:`\pi_i = \dfrac{1}{1+e^{-\pmb y_i \pmb\beta}}`

Manual and Examples
==========================
  - Manual https://github.com/maxdevblock/cubmods/blob/main/Manual/02_cub_family.md

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

Credits
==============
    :Author:      Massimo Pierini
    :Institution: Universitas Mercatorum
    :Affiliation: Graduand in Statistics & Big Data (L41)
    :Date:        2023-24
    :Credit:      Domenico Piccolo, Rosaria Simone
    :Contacts:    cub@maxpierini.it

Classes and Functions
=====================
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
    r"""Probability distribution for each subject of a specified CUB model 
    with covariates.
    
    Auxiliary function of ``.draw()``.

    :math:`\Pr(R = r | \pmb\theta_i ; \pmb y_i),\; i=1 \ldots n ,\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :return: the matrix of the probability distribution of dimension :math:`n \times r`
    :rtype: numpy ndarray
    """
    n = Y.shape[0]
    pi_i = logis(Y, beta)
    p = np.ndarray(shape=(n,m))
    for i in range(n):
        p[i,:] = cub_pmf(m=m, pi=pi_i[i],
            xi=xi)
    return p

def pmf(m, beta, xi, Y):
    r"""Average probability distribution of a specified CUB model 
    with covariates.

    :math:`\frac{1}{n} \sum_{i=1}^n \Pr(R = r | \pmb\theta_i ; \pmb w_i),\; r=1 \ldots m`

    :param m: number of ordinal categories
    :type m: int
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :return: the vector of the probability distribution.
    :rtype: numpy array
    """
    p = pmfi(m, beta, xi, Y)
    pr = p.mean(axis=0)
    return pr

def prob(m, sample, Y, beta, xi):
    r"""Probability distribution of a CUB model with covariates for the uncertainty component
    given an observed sample

    Compute the probability distribution of a CUB model with covariates
    for the feeling component, given an observed sample.
    
    :math:`\Pr(R = r_i | \pmb\theta_i ; \pmb w_i),\; i=1 \ldots n`
    
    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param xi: uncertainty parameter :math:`\xi`
    :type xi: float
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :return: the array of the probability distribution.
    :rtype: numpy array
    """
    p = (
        logis(Y=Y, param=beta)*
        (bitxi(m=m, sample=sample, xi=xi) - 1/m)
        ) + 1/m
    return p

def loglik(m, sample, Y, beta, xi):
    r"""Log-likelihood function of a CUB model with covariates for the uncertainty component

    Compute the log-likelihood function of a CUB model fitting ordinal responses with covariates 
    for explaining the uncertainty component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param xi: uncertainty parameter :math:`\xi`
    :type xi: float
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :return: the log-likelihood value
    :rtype: float
    """
    p = probbit(m, xi)
    pn = p[sample-1]
    eta = logis(Y, param=beta)
    l = np.sum(np.log(eta*(pn-1/m)+1/m))
    return l

def draw(m, beta, xi, Y,
    df, formula, seed=None):
    r"""Draw a random sample from a specified CUB model with covariates for
    the uncertainty component.

    :param m: number of ordinal categories
    :type m: int
    :param n: number of ordinal responses to be drawn
    :type n: int
    :param xi: uncertainty parameter :math:`\xi`
    :type xi: float
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param seed: the `seed` to ensure reproducibility, defaults to None;
        it must be :math:`\neq 0`
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    """
    #np.random.seed(seed)
    assert len(beta) == Y.shape[1]+1
    n = Y.shape[0]
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
        seed=seed, diss=diss, df=df,
        theoric=theoric, formula=formula
    )
    return sample

def varcov(m, sample, Y, beta, xi):
    r"""Variance-covariance matrix of CUB model with covariates for the uncertainty parameter.

    Compute the variance-covariance matrix of parameter estimates of a CUB model with 
    covariates for the uncertainty component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param xi: uncertainty parameter :math:`\xi`
    :type xi: float
    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :return: the variance-covariance matrix of the CUB model
    :rtype: numpy ndarray
    """
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
    r"""Auxiliary function for the log-likelihood estimation of CUB models.

    Compute the opposite of the scalar function that is maximized when running
    the E-M algorithm for CUB models with covariates for the uncertainty parameter. 

    It is called as an argument for optim within CUB function for models with covariates for
        uncertainty or for both feeling and uncertainty.

    :param beta: array :math:`\pmb \beta` of parameters for the uncertainty component, whose length equals 
        ``Y.columns.size+1`` to include an intercept term in the model (first entry)
    :type beta: array of float
    :param esterno10: A matrix binding together: the matrix :math:`\pmb y` of the selected covariates  
        (accounting for an intercept term) and a vector :math:`\tau` (whose length equals the number of observations) 
        of the posterior probabilities that each observation has been generated by the first component 
        distribution of the mixture
    :return: the expected value of the inconplete log-likelihood
    :rtype: float
    """
    tauno = esterno10[:,0]
    covar = esterno10[:,1:]
    covbet = covar @ beta
    r = np.sum(
        np.log(1+np.exp(-covbet))
        +(1-tauno)*covbet
    )
    return r

def mle(sample, m, Y, df, formula,
    gen_pars=None,
    maxiter=500,
    tol=1e-4):
    r"""
    Main function for CUB models with covariates for the uncertainty component.

    Estimate and validate a CUB model for given ordinal responses, with covariates for explaining 
    the uncertainty component.

    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param Y: dataframe of covariates for explaining the uncertainty component;
        no column must be named ``0`` nor ``constant``
    :type Y: pandas dataframe
    :param df: original DataFrame
    :type df: DataFrame
    :param formula: the formula used
    :type formula: str
    :param gen_pars: dictionary of hypothesized parameters, defaults to None
    :type gen_pars: dictionary, optional
    :param maxiter: maximum number of iterations allowed for running the optimization algorithm
    :type maxiter: int
    :param tol: fixed error tolerance for final estimates
    :type tol: float
    :return: an instance of ``CUBresCUBY0`` (see the Class for details)
    :rtype: object
    """
    # start datetime
    start = dt.datetime.now()
    # cast sample to numpy array
    sample = np.array(sample)
    # model ordinal categories
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
            diss=diss, df=df,
            formula=formula,
            gen_pars=gen_pars
            # pi_gen=pi_gen, xi_gen=xi_gen
        )
    return res

class CUBresCUBY0(CUBres):
    r"""Object returned by ``.mle()`` function.
    See the Base for details.
    """

    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None, kind="bar",
        saveas=None
        ):
        r"""Plots relative average frequencies of observed sample, estimated average probability distribution and,
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
        title = "AVERAGE ESTIMATED PROBABILITY\n"
        title += f"{self.model} model    "
        title += f"$n={self.n}$\n"
        title += f"    Dissim(est,obs)={self.diss:.4f}"
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
        """Main function to plot an object of the Class.

        :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
        :type figsize: tuple of float
        :param saveas: if provided, name of the file to save the plot
        :type saveas: str
        :return: ``ax`` or a tuple ``(fig, ax)``
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_ordinal(ax=ax)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
    