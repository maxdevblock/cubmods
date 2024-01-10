"""
CUB models in Python.
Module for General functions.

Description:
    This module contains methods and classes
    for general functions.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    from cubmods import general

    R = choices(m=m)

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
# pylint: disable=anomalous-backslash-in-string
import re
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
import scipy.stats as sps
from scipy.special import binom
from matplotlib.patches import Ellipse
from matplotlib import transforms

#TODO anytime a function is called, use explicit kwargs!!!
def choices(m):
    """
    preference choices of CUB model
    """
    return np.arange(m)+1

def probbit(m, xi):
    R = choices(m)
    p = sps.binom(n=m-1, p=1-xi).pmf(R-1)
    return p

def freq(sample, m, dataframe=False):
    """
    absolute frequecies of CUB sample
    """
    f = []
    R = choices(m)
    for r in R:
        f.append(sample[sample==r].size)
    f = np.array(f)
    if not dataframe:
        return f
    df = pd.DataFrame({
        "choice": R,
        "freq": f
    }).set_index("choice")
    return df
    
def chisquared(f_obs, f_exp):
    """
    compute chi-squared
    """
    cont = f_obs - f_exp
    return np.sum(cont**2 / f_exp)
    
def dissimilarity(p_obs, p_est):
    """
    compute dissimilarity index
    """
    return np.sum(abs(p_obs-p_est))/2

def colsof(A):
    shape = A.shape
    if len(shape) == 1:
        return 1
    else:
        return shape[1]

def addones(A):
    AA = np.c_[np.ones(A.shape[0]), A]
    return AA

def bic(l, p, n):
    return -2*l + np.log(n)*p

def aic(l, p):
    return -2*l + 2*p

def luni(m, n):
    # loglik of null model (uniform)
    loglikuni = -(n*np.log(m))
    return loglikuni

#TODO: remove unused argument m and modify in all modules
def lsat(m, f, n):
    # loglik of saturated model
    logliksat = -(n*np.log(n)) + np.sum((f[f!=0])*np.log(f[f!=0]))
    return logliksat

def lsatcov(sample, covars):
    df = pd.DataFrame({"ord":sample}).join(
        covars)
    #TODO: solve overlapping cols if same cov for more pars
    cov = list(df.columns[1:])
    logliksatcov = np.sum(
        np.log(
        df.value_counts().div(
        df[cov].value_counts())))
    return logliksatcov

def kkk(sample, m):
    #' @title Sequence of combinatorial coefficients
    #' @description Compute the sequence of binomial coefficients \eqn{{m-1}\choose{r-1}}, for \eqn{r= 1, \dots, m}, 
    #' and then returns a vector of the same length as ordinal, whose i-th component is the corresponding binomial 
    #' coefficient \eqn{{m-1}\choose{r_i-1}}
    #' @aliases kkk
    #' @keywords internal
    #' @usage kkk(m, ordinal)
    #' @param m Number of ordinal categories
    #' @param Vector of ordinal responses
    R = choices(m)
    v = binom(m-1, R-1)
    return v[sample-1]

def logis(Y, param):
    #' @title The logistic transform
    #' @description Create a matrix YY binding array \code{Y} with a vector of ones, placed as the first column of YY. 
    #' It applies the logistic transform componentwise to the standard matrix multiplication between YY and \code{param}.
    #' @aliases logis
    #' @usage logis(Y,param)
    #' @export logis
    #' @param Y A generic matrix or one dimensional array
    #' @param param Vector of coefficients, whose length is NCOL(Y) + 1 (to consider also an intercept term)
    #' @return Return a vector whose length is NROW(Y) and whose i-th component is the logistic function
    #' at the scalar product between the i-th row of YY and the vector \code{param}.
    #' @keywords utilities
    #' @examples
    #' n<-50 
    #' Y<-sample(c(1,2,3),n,replace=TRUE) 
    #' param<-c(0.2,0.7)
    #' logis(Y,param)
    YY = np.c_[np.ones(Y.shape[0]), Y]
    val = 1/(1 + np.exp(-YY @ param))
    #TODO: implement if (all(dim(val)==c(1,1)))
    return val

def bitgamma(sample, m, W, gamma):
    #' @title Shifted Binomial distribution with covariates
    #' @description Return the shifted Binomial probabilities of ordinal responses where the feeling component 
    #' is explained by covariates via a logistic link.
    #' @aliases bitgama
    #' @usage bitgama(m,ordinal,W,gama)
    #' @param m Number of ordinal categories
    #' @param ordinal Vector of ordinal responses
    #' @param W Matrix of covariates for the feeling component
    #' @param gama Vector of parameters for the feeling component, with length equal to 
    #' NCOL(W)+1 to account for an intercept term (first entry of \code{gama})
    #' @export bitgama
    #' @return A vector of the same length as \code{ordinal}, where each entry is the shifted Binomial probability for
    #'  the corresponding observation and feeling value.
    #' @seealso  \code{\link{logis}}, \code{\link{probcub0q}}, \code{\link{probcubpq}} 
    #' @keywords distribution
    #' @import stats
    #' @examples 
    #' n<-100
    #' m<-7
    #' W<-sample(c(0,1),n,replace=TRUE)
    #' gama<-c(0.2,-0.2)
    #' csivett<-logis(W,gama)
    #' ordinal<-rbinom(n,m-1,csivett)+1
    #' pr<-bitgama(m,ordinal,W,gama)
    ci = 1/logis(Y=W, param=gamma) - 1
    bg = kkk(sample=sample, m=m) * np.exp(
        (sample-1)*np.log(ci)-(m-1)*np.log(1+ci))
    return bg
    
def bitxi(m, sample, xi):
    base = np.log(1-xi)-np.log(xi)
    cons = np.exp(m*np.log(xi)-np.log(1-xi))
    cons *= kkk(sample=sample, m=m)*np.exp(base*sample)
    return cons

def hadprod(Amat, xvett):
    ra = Amat.shape[0]
    ca = Amat.shape[1]
    dprod = np.zeros(shape=(ra, ca))
    for i in range(ra):
        dprod[i,:] = Amat[i,:] * xvett[i]
    return dprod

def conf_ell(vcov, mux, muy, ci, ax, showaxis=True):
    """
    plot confidence ellipse of estimated
    CUB parameters of ci% on ax
    """
    nstd = sps.norm().ppf((1-ci)/2)
    rho = vcov[0,1]/np.sqrt(vcov[0,0]*vcov[1,1])
    # beta1 = vcov[0,1] / vcov[0,0]
    radx = np.sqrt(1+rho)
    rady = np.sqrt(1-rho)
    ell = Ellipse(
        (0,0), 2*radx, 2*rady,
        color="b", alpha=.25,
        label=f"CR {ci:.0%}"
    )
    scale_x = np.sqrt(vcov[0, 0]) * nstd
    scale_y = np.sqrt(vcov[1, 1]) * nstd
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mux, muy)
    ell.set_transform(transf + ax.transData)
    elt = ax.add_patch(ell)
    # ang = elt.get_angle()
    # ax.axline([mux, muy], slope=ang)
    # if showaxis:
    #     ax.annotate("",
    #         xy=(ell.center[0] - ell.width +2,
    #             ell.center[1] - ell.height ),
    #         xytext=(ell.center[0] + ell.width-1,
    #                 ell.center[1] + ell.height+2),
    #         arrowprops=dict(arrowstyle="<->", color="red"),
    #         transform=transf
    #     )

def load_object(fname):
    """
    Load a saved object from file
    """
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    print(f"Object `{obj}` loaded from {fname}")
    return obj

#TODO: test in GEM
def formula_parser(formula):
    if '~' not in formula:
        print("ERR: ~ missing")
    if formula.count("|") != 2:
        print("ERR: | must be 2")
    regex = "^([a-zA-Z0-9_()]{1,})~"
    regex += "([a-zA-Z0-9_+()]{1,})\|"
    regex += "([a-zA-Z0-9_+()]{1,})\|"
    regex += "([a-zA-Z0-9_+()]{1,})$"
    if not re.match(regex, formula):
        print("ERR: wrong formula")
        return None
    # split y from X
    yX = formula.split('~')
    # define y
    y = yX[0]
    # split all X
    X = yX[1].split("|")
    # prepare matrix
    XX = []
    for x in X:
        if x == "0":
            XX.append(None)
            continue
        x = x.split("+")
        XX.append(x)
    return y, XX
    
def dummies(df, DD):
    # new covars
    XX = []
    for D in DD:
        if D is None:
            XX.append(None)
            continue
        X = []
        for d in D:
            if d[:2]=="C(" and d[-1]==")":
                c = d[2:-1]
                # str to avoid floats
                if is_float_dtype(df[c]):
                    df[c] = df[c].astype(int)
                df = pd.get_dummies(
                    df, columns=[c],
                    drop_first=True,
                    prefix=f"C.{c}"
                )
                # dummies names
                f = [f"C.{c}_" in i for i in df.columns]
                dums = df.columns[f]
                for dum in dums:
                    X.append(dum)
            else:
                X.append(d)
        XX.append(X)
    return df, XX

###########################################
# EXCEPTION ERROR CLASSES
###########################################

class InvalidCategoriesError(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, m, model):
        self.m = m
        self.model = model
        self.msg = f"Insufficient categories {self.m} for model {self.model}"
        super().__init__(self.msg)

class UnknownModelError(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, model):
        self.model = model
        self.msg = f"Unknown model {self.model}"
        super().__init__(self.msg)

class NotImplementedModelError(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, model, formula):
        self.formula = formula
        self.model = model
        self.msg = f"Not 8mplemented model {self.model} with formula {self.formula}"
        super().__init__(self.msg)

class NoShelterError(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, model):
        self.model = model
        self.msg = f"Shelter choice (sh) needed for {self.model} model"
        super().__init__(self.msg)

#TODO: add in draw & mle in cubsh, cush
class ShelterGreaterThanM(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, m, sh):
        self.m = m
        self.sh = sh
        self.msg = f"Shelter choice must be in [1,m], given sh={self.sh} with m={self.m}"
        super().__init__(self.msg)

#TODO: add in all draw
class ParameterOurOfBoundsError(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, param, value):
        self.param = param
        self.value = value
        self.msg = f"{self.value} is out of bounds for parameter {self.param}"
        super().__init__(self.msg)

#TODO: add in all draw
class InvalidSampleSizeError(Exception):
    """
    if m is not suitable for model
    """
    def __init__(self, n):
        self.n = n
        self.msg = f"Sample size must be strictly > 0, given {self.n}"
        super().__init__(self.msg)
