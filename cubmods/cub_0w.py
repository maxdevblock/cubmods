# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements
#TODO: go on with implementation
import datetime as dt
import pickle
import numpy as np
import pandas as pd
#from scipy.special import binom
from scipy.optimize import minimize
import scipy.stats as sps
import matplotlib.pyplot as plt
from .general import (
    choices, freq, dissimilarity,
    chisquared, conf_ell, bitgamma,
    logis, hadprod, luni, lsat,
    lsatcov, addones, colsof, aic, bic
)
from . import cub
from .smry import CUBres, CUBsample

###################################################################
# FUNCTIONS
###################################################################

def pmf(m, pi, gamma, W):
    n = W.shape[0]
    p = pmfi(m, pi, gamma, W)
    pr = p.mean(axis=0)
    return pr

def pmfi(m, pi, gamma, W):
    n = W.shape[0]
    xi_i = logis(W, gamma)
    p = np.ndarray(shape=(n, m))
    for i in range(n):
        xi = xi_i[i]
        p[i,:] = cub.pmf(m=m, pi=pi, xi=xi)
    #pr = p.mean(axis=0)
    return p

def prob(sample, m, pi, gamma, W):
    #' @title Probability distribution of a CUB model with covariates for the feeling component
    #' @aliases probcub0q
    #' @description Compute the probability distribution of a CUB model with covariates
    #'  for the feeling component.
    #' @export probcub0q
    #' @usage probcub0q(m,ordinal,W,pai,gama)
    #' @keywords distribution
    #' @param m Number of ordinal categories
    #' @param ordinal Vector of ordinal responses 
    #' @param W Matrix of covariates for explaining the feeling component
    #' NCOL(Y)+1 to include an intercept term in the model (first entry)
    #' @param pai Uncertainty parameter
    #' @param gama Vector of parameters for the feeling component, whose length equals 
    #' NCOL(W)+1 to include an intercept term in the model (first entry)
    #' @return A vector of the same length as \code{ordinal}, whose i-th component is the
    #' probability of the i-th observation according to a CUB distribution with the corresponding values 
    #' of the covariates for the feeling component and coefficients specified in \code{gama}.
    #' @seealso \code{\link{bitgama}}, \code{\link{probcub00}}, \code{\link{probcubp0}}, 
    #' \code{\link{probcubpq}}
    #' @references 
    #' Piccolo D. (2006). Observed Information Matrix for MUB Models, 
    #' \emph{Quaderni di Statistica}, \bold{8}, 33--78 \cr
    #' Piccolo D. and D'Elia A. (2008). A new approach for modelling consumers' preferences, \emph{Food Quality and Preference},
    #' \bold{18}, 247--259 \cr
    #' Iannario M. and Piccolo D. (2012). CUB models: Statistical methods and empirical evidence, in: 
    #' Kenett R. S. and Salini S. (eds.), \emph{Modern Analysis of Customer Surveys: with applications using R}, 
    #' J. Wiley and Sons, Chichester, 231--258
    #' @examples
    #' data(relgoods)
    #' m<-10
    #' naord<-which(is.na(relgoods$Physician))
    #' nacov<-which(is.na(relgoods$Gender))
    #' na<-union(naord,nacov)
    #' ordinal<-relgoods$Physician[-na]
    #' W<-relgoods$Gender[-na]
    #' pai<-0.44; gama<-c(-0.91,-0.7)
    #' pr<-probcub0q(m,ordinal,W,pai,gama)
    """
    PMF of CUB model with covariates for xi
    """
    p = pi*(bitgamma(sample=sample, m=m, W=W, gamma=gamma)-1/m) + 1/m
    return p

def proba(m, pi, xi, r): #TODO
    """
    probability Pr(R=r) of CUB model
    """
    return None

def cmf(sample, m, pi, gamma, W): #TODO: test
    """
    CMF of CUB model
    """
    return prob(sample, m, pi, gamma, W).cumsum()

def mean(m, pi, xi): #TODO
    """
    mean of CUB model
    """
    return None

def var(m, pi, xi): #TODO
    """
    variance of CUB model
    """
    return None

def std(m, pi, xi): #TODO
    """
    standard deviation of CUB model
    """
    return None

def skew(pi, xi): #TODO
    """
    skewness normalized eta index
    """
    return None

def mean_diff(m, pi, xi): #TODO
    return None
    
def median(m, pi, xi): #TODO
    return None
    
def gini(m, pi, xi): #TODO
    return None
    
def laakso(m, pi, xi): #TODO
    return None

def rvs(m, pi, gamma, W): #TODO
    """
    generate random sample from CUB model
    """
    n = W.shape[0]
    rv = np.random.choice(
        choices(m=m),
        size=n,
        replace=True,
        p=pmf(m, pi, gamma, W)
        )
    return rv

def loglik(sample, m, pi, gamma, W): #TODO: test
    #' @title Log-likelihood function of a CUB model with covariates for the feeling component
    #' @description Compute the log-likelihood function of a CUB model fitting ordinal data, with \eqn{q} 
    #' covariates for explaining the feeling component.
    #' @aliases loglikcub0q
    #' @usage loglikcub0q(m, ordinal, W, pai, gama)
    #' @param m Number of ordinal categories
    #' @param ordinal Vector of ordinal responses
    #' @param W Matrix of selected covariates for explaining the feeling component
    #' @param pai Uncertainty parameter
    #' @param gama Vector of parameters for the feeling component, with length NCOL(W) + 1 to account for 
    #' an intercept term (first entry of gama)
    #' @keywords internal
    p = prob(sample, m, pi, gamma, W)
    l = np.sum(np.log(p))
    return l

def varcov(sample, m, pi, gamma, W):
    #' @title Variance-covariance matrix of CUB models with covariates for the feeling component
    #' @description Compute the variance-covariance matrix of parameter estimates of a CUB model
    #'  with covariates for the feeling component.
    #' @aliases varcovcub0q
    #' @usage varcovcub0q(m, ordinal, W, pai, gama)
    #' @param m Number of ordinal categories
    #' @param ordinal Vector of ordinal responses
    #' @param W Matrix of covariates for explaining the feeling component
    #' @param pai Uncertainty parameter
    #' @param gama Vector of parameters for the feeling component, whose length is 
    #' NCOL(W)+1 to include an intercept term in the model (first entry of gama)
    #' @export varcovcub0q
    #' @details The function checks if the variance-covariance matrix is positive-definite: if not, 
    #' it returns a warning message and produces a matrix with NA entries.
    #' @keywords internal
    #' @references
    #' Piccolo D.(2006), Observed Information Matrix for MUB Models. \emph{Quaderni di Statistica},
    #'  \bold{8}, 33--78,
    #' @examples
    #' data(univer)
    #' m<-7
    #' ordinal<-univer[,9]
    #' pai<-0.86
    #' gama<-c(-1.94, -0.17)
    #' W<-univer[,4]           
    #' varmat<-varcovcub0q(m, ordinal, W, pai, gama)

    """
    compute asymptotic variance-covariance
    of CUB estimated parameters
    """
    qi = 1/(m*prob(sample,m,pi,gamma,W))
    qistar = 1 - (1-pi)*qi
    qitilde = qistar*(1-qistar)
    fi = logis(W, gamma)
    fitilde = fi*(1-fi)
    ai = (sample-1) - (m-1)*(1-fi)
    g01 = (ai*qi*qistar)/pi
    hh = (m-1)*qistar*fitilde - (ai**2)*qitilde
    WW = addones(W)
    i11 = np.sum((1-qi)**2 / pi**2)
    i12 = g01.T @ WW
    i22 = WW.T @ hadprod(WW, hh) #TODO: check if this is Hadarmad or not
    # Information matrix
    nparam = colsof(WW) + 1
    matinf = np.ndarray(shape=(nparam, nparam))
    matinf[:] = np.nan
    matinf[0,:] = np.concatenate([[i11], i12]).T #TODO: check dimensions

    varmat = np.ndarray(shape=(nparam, nparam))
    varmat[:] = np.nan
    for i in range(1, nparam):
        matinf[i,:] = np.concatenate([
            [i12[i-1]], i22[i-1,:]]).T #TODO: check dimensions
    if np.any(np.isnan(matinf)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(matinf) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        varmat = np.linalg.inv(matinf)
    return varmat

def init_gamma(sample, m, W): #TODO test
    WW = np.c_[np.ones(W.shape[0]), W]
    ni = np.log((m-sample+.5)/(sample-.5))
    #TODO: .solve or .inv ????
    gamma = np.linalg.inv(WW.T @ WW) @ (WW.T @ ni)
    return gamma

###################################################################
# RANDOM SAMPLE
###################################################################

def draw(m, n, pi, gamma, W, seed=None): #TODO
    """
    generate random sample from CUB model
    """
    #np.random.seed(seed)
    assert n == W.shape[0]
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, pi=pi,
        gamma=gamma, W=W)
    #print("n", n)
    for i in range(n):
        #TODO: if seed is not None
        if seed is not None:
            np.random.seed(seed*i)
        rv[i] = np.random.choice(
            choices(m=m),
            size=1,
            replace=True,
            p=theoric_i[i]
        )
    f = freq(m=m, sample=rv)
    theoric = pmf(m=m,pi=pi,gamma=gamma,W=W)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        [pi], gamma
    ))
    par_names = np.concatenate((
        ["pi"],
        ["constant"],
        W.columns
    ))
    sample = CUBsample(
        model="CUB(0W)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, W=W, diss=diss,
        theoric=theoric
    )
    return sample

###################################################################
# INFERENCE
###################################################################
def effe01(gamma, esterno01, m): #TODO test
    #' @title Auxiliary function for the log-likelihood estimation of CUB models
    #' @description Compute the opposite of the scalar function that is maximized when running 
    #' the E-M algorithm for CUB models with covariates for the feeling parameter.
    #' @aliases effe01
    #' @usage effe01(gama, esterno01, m)
    #' @param gama Vector of the starting values of the parameters to be estimated
    #' @param esterno01 A matrix binding together the vector of the posterior probabilities
    #' that each observation has been generated by the first component distribution of the mixture, 
    #' the ordinal data and the matrix of the selected covariates accounting for an intercept term
    #' @keywords internal 
    #' @details It is called as an argument for optim within CUB function for models with covariates for
    #' feeling or for both feeling and uncertainty
    ttau = esterno01[:,0]
    ordd = esterno01[:,1]
    covar = esterno01[:,2:]
    covar_gamma = covar @ gamma
    r = np.sum(
        ttau*(
            (ordd-1)*(covar_gamma)
            +
            (m-1)*np.log(1+np.exp(-covar_gamma))
        )
    )
    return r

def mle(sample, m, W, #TODO
    gen_pars=None,
    maxiter=500,
    tol=1e-4,
    ci=.99):
    """
    fit a sample to a CUB model
    with m preference choices.
    if the sample has been generated
    from a CUB model itself and
    generating (pi, xi) are known,
    compute compare metrics
    """
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
    #TODO: use this?
    aver = np.mean(sample)
    # add a column of 1
    WW = addones(W)
    # number of covariates
    q = colsof(W)
    # initialize gamma parameter
    gammajj = init_gamma(sample=sample, m=m, W=W)
    # initialize (pi, xi)
    pijj, _ = cub.init_theta(f=f, m=m)
    # compute loglikelihood
    l = loglik(sample, m, pijj, gammajj, W)

    # start E-M algorithm
    niter = 1
    while niter < maxiter:
        lold = l
        vettn = bitgamma(sample=sample, m=m, W=W, gamma=gammajj)
        ttau = 1/(1+(1-pijj)/(m*pijj*vettn))
        #print(f"niter {niter} ***************")
        #print("vettn")
        #print(vettn)
        #print("ttau")
        #print(ttau)
        ################################# maximize w.r.t. gama  ########
        esterno01 = np.c_[ttau, sample, WW]
        optimgamma = minimize(
            effe01, x0=gammajj, args=(esterno01, m),
            method="Nelder-Mead"
            #method="BFGS"
        )
        #print(optimgamma)
        ################################################################
        gammajj = optimgamma.x #[0]
        #print(f"gama {gammajj}")
        pijj = np.sum(ttau)/n
        l = loglik(sample, m, pijj, gammajj, W)
        # compute delta-loglik
        deltal = abs(l-lold)
        # check tolerance
        if deltal <= tol:
            break
        else:
            lold = l
        niter += 1
    # end E-M algorithm
    pi = pijj
    gamma = gammajj
    #l = loglikjj
    # variance-covariance matrix
    varmat = varcov(sample, m, pi, gamma, W)
    end = dt.datetime.now()

    # Akaike Information Criterion
    AIC = aic(l=l, p=q+2)
    # Bayesian Information Criterion
    BIC = bic(l=l, p=q+2, n=n)

    #print(pi)
    #print(gamma)
    #print(niter)
    #print(l)
    #return None

    # standard errors
    stderrs = np.sqrt(np.diag(varmat))

    #print(stderrs)
    #return None
    # Wald statistics
    wald = np.concatenate([[pi], gamma])/stderrs
    #print(wald)
    #return None
    # p-value
    pval = 2*(sps.norm().sf(abs(wald)))
    # mean loglikelihood
    muloglik = l/n
    # loglik of null model (uniform)
    loglikuni = luni(n=n, m=m)
    # loglik of saturated model
    logliksat = lsat(f=f, n=n)
    #TODO: TEST LOGLIK SAT FOR COVARIATES
    #      see https://stackoverflow.com/questions/77791392/proportion-of-each-unique-value-of-a-chosen-column-for-each-unique-combination-o#77791442
    #df = pd.merge(
    #    pd.DataFrame({"ord":sample}),
    #    W,
    #    left_index=True, right_index=True
    #)
    #df = pd.DataFrame({"ord":sample}).join(W)
    #cov = list(W.columns)
    #logliksatcov = np.sum(
    #    np.log(
    #    df.value_counts().div(
    #    df[cov].value_counts())))
    #logliksatcov = lsatcov(
    #    sample=sample,
    #    covars=[W]
    #)
    # loglik of shiftet binomial
    # xibin = (m-sample.mean())/(m-1)
    # loglikbin = loglik(m, 1, xibin, f)
    # Explicative powers
    # Ebin = (loglikbin-loglikuni)/(logliksat-loglikuni)
    # Ecub = (l-loglikbin)/(logliksat-loglikuni)
    # Ecub0 = (l-loglikuni)/(logliksat-loglikuni)
    # deviance from saturated model
    dev = 2*(logliksat-l)
    # ICOMP metrics
    #npars = q
    #trvarmat = np.sum(np.diag(varmat))
    #ICOMP = -2*l + npars*np.log(trvarmat/npars) - np.log(np.linalg.det(varmat))
    # coefficient of correlation
    # rho = varmat[0,1]/np.sqrt(varmat[0,0]*varmat[1,1])
    theoric = pmf(m=m, pi=pi, gamma=gamma, W=W)
    diss = dissimilarity(f/n, theoric)
    gamma_names = np.concatenate([
        ["constant"],
        W.columns])
    estimates = np.concatenate((
        [pi], gamma
    ))
    est_names = np.concatenate((
        ["pi"], gamma_names
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        ["Feeling"],
        np.repeat(None, q)
    ))
    # compare with known (pi, xi)
    # if pi_gen is not None and xi_gen is not None:
    #     pass
    # results object
    res = CUBresCUB0W(
            model="CUB(0W)",
            m=m, n=n, niter=niter,
            maxiter=maxiter, tol=tol,
            estimates=estimates,
            est_names=est_names,
            e_types=e_types,
            stderrs=stderrs,
            pval=pval, wald=wald,
            loglike=l, muloglik=muloglik,
            loglikuni=loglikuni,
            logliksat=logliksat,
            #logliksatcov=logliksatcov,
            # loglikbin=loglikbin,
            # Ebin=Ebin, Ecub=Ecub, Ecub0=Ecub0,
            theoric=theoric,
            dev=dev,
            AIC=AIC, BIC=BIC,
            seconds=(end-start).total_seconds(),
            time_exe=start,
            # rho=rho,
            sample=sample, f=f,
            varmat=varmat,
            W=W,
            diss=diss,
            # pi_gen=pi_gen, xi_gen=xi_gen
        )
    return res

class CUBresCUB0W(CUBres): #TODO

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

    def save(self, fname):
        """
        Save a CUBresult object to file
        """
        filename = f"{fname}.cub.fit"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Fitting saved to {filename}")