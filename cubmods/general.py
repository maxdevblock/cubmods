# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
"""
CUB models in Python.
Module for General functions.

Description:
============
    This module contains methods and classes
    for general functions.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

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

:Author:      Massimo Pierini
:Institution: Universitas Mercatorum
:Affiliation: Graduand in Statistics & Big Data (L41)
:Date:        2023-24
:Credit:      Domenico Piccolo, Rosaria Simone
:Contacts:    cub@maxpierini.it

Classes and Functions
=====================
"""

import re
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
import scipy.stats as sps
from scipy.special import binom
from scipy.linalg import sqrtm
from matplotlib.patches import Ellipse
from matplotlib import transforms
#from .cub import loglik as lcub

def choices(m):
    """Array of ordinal categories.
    
    :param m: number of ordinal categories
    :type m: int
    :return: array of int from 1 to m
    :rtype: array
    """
    return np.arange(m)+1

def probbit(m, xi):
    r"""Probability distribution of 
    shifted binomial random variable.
    
    :param m: number of ordinal categories
    :type m: int
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: the vector of the probability 
        distribution of a shifted Binomial 
        model.
    :rtype: array
    """
    R = choices(m)
    p = sps.binom(n=m-1, p=1-xi).pmf(R-1)
    return p

def freq(sample, m, dataframe=False):
    """Absolute frequecies of an
    observed sample of ordinal
    responses.
    
    :param sample: array of ordinal responses
    :type sample: array of int
    :param m: number of ordinal categories
    :type m: int
    :param dataframe: if ``True`` return
        a DataFrame instead of an array,
        defaults to ``False``
    :type dataframe: bool
    :return: the absolute frequencies of the observed sample
    :rtype: array or dataframe
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
    
def _chisquared(f_obs, f_exp):
    """
    compute chi-squared
    """
    cont = f_obs - f_exp
    return np.sum(cont**2 / f_exp)
    
def dissimilarity(p_obs, p_est):
    """Normalized dissimilarity measure.
    
    Compute the normalized dissimilarity measure between observed
    relative frequencies and estimated (theoretical) probabilities of a discrete distribution.
    
    :param p_obs: Vector of observed relative frequencies
    :type p_obs: array
    :param p_est: Vector of estimated (theoretical) probabilities
    :type p_est: array
    :return: Numeric value of the dissimilarity index, assessing the distance to a perfect fit.
    :rtype: float
    """
    return np.sum(abs(p_obs-p_est))/2

def colsof(A):
    r"""Number of columns of the given
    matrix or dataframe.
    
    :param A: the matrix or dataframe
    :type A: ndarray, dataframe
    :return: number of columns
    :rtype: int
    """
    shape = A.shape
    if len(shape) == 1:
        return 1
    else:
        return shape[1]

def addones(A):
    r"""Expand with a unitary vector in the first column of the given matrix
    to consider also an intercept term for CUB models with covariates.

    :param A: a matrix to be expanded
    :type a: ndarray or DataFrame
    :return: the expanded matrix
    :rtype: same of ``A``
    """
    AA = np.c_[np.ones(A.shape[0]), A]
    return AA

def bic(l, p, n):
    r"""Bayesian Information Criterion.
    
    :param l: log-likelihood
    :type l: float
    :param p: number of parameters
    :type p: int
    :param n: number of observations
    :type n: int
    :return: the BIC value
    :rtype: float
    """
    return -2*l + np.log(n)*p

def aic(l, p):
    r"""Akaike Information Criterion.
    
    :param l: log-likelihood
    :type l: float
    :param p: number of parameters
    :type p: int
    :return: the AIC value
    :rtype: float
    """
    return -2*l + 2*p

def luni(m, n):
    r"""Log-likelihood of null model.
    
    Null level, that is when no 
    structure is searched for. 
    Specifically, this is equivalent to 
    assume a discrete Uniform over 
    the support so that any category 
    has the same probability. 
    
    :param m: number of ordinal categories
    :type m: int
    :param n: number of observations
    :type n: int
    :return: the log-likelihood of null model
    :rtype: float
    """
    loglikuni = -(n*np.log(m))
    return loglikuni

def lsat(f, n):
    r"""Log-likelihood of saturated model.

    Ssaturated level,that is the theoretically maximum information
    that can be obtained by a model using as many parameters as possible. 
    Then, the saturated log-likelihood is computed by assuming that the model 
    is specified by as many parameters as available observations. 
    This is the extreme benchmark for comparing
    previous log-likelihood quantities.

    :param f: absolute frequencies of observed ordinal responses
    :type f: array
    :param n: number of observations
    :type n: int
    :return: log-likelihood of saturated model
    :rtype: float
    """
    # loglik of saturated model
    logliksat = -(n*np.log(n)) + np.sum((f[f!=0])*np.log(f[f!=0]))
    return logliksat

#TODO: add loglikbin to all models and smry ???
def _lbin(sample, m, f):
    r"""Log-likelihood of shifted Binomial model.
    """
    avg = sample.mean()
    xi = (m-avg)/(m-1)
    R = choices(m)
    p = binom(m-1, R-1) * (1-xi)**(R-1) * xi**(m-R)
    l = np.sum(f*np.log(p))
    return l

#TODO: is lsatcov useful?
def _lsatcov(sample, covars):
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
    r"""Sequence of combinatorial coefficients

    Compute the sequence of binomial coefficients :math:`\binom{m-1}{r-1}`, for :math:`r= 1, \ldots m`, 
    and then returns a vector of the same length as ordinal, whose i-th component is the corresponding binomial 
    coefficient :math:`\binom{m-1}{r_i-1}`

    :param sample: array of ordinal responses
    :type sample: array
    :param m: number of ordinal categories
    :type m: int
    :return: an array of :math:`\binom{m-1}{r_i-1}`
    :rtype: array
    """
    R = choices(m)
    v = binom(m-1, R-1)
    return v[sample-1]

def logis(Y, param):
    r"""The logistic transform.

    Create a matrix ``YY`` binding array ``Y`` with a vector of ones, placed as the first column of ``YY``. 
    It applies the logistic transform componentwise to the standard matrix multiplication between ``YY`` and ``param``.

    :param Y: A generic matrix or a dataframe
    :type Y: ndarray, dataframe
    :param param: Vector of coefficients, whose length is ``Y.columns.size+1`` (to consider also an intercept term)
    :type param: array
    :return: a vector whose length is ``Y.index.size`` and whose i-th component is the logistic function
    """
    YY = np.c_[np.ones(Y.shape[0]), Y]
    YY = YY.astype(float)
    #print(YY, param)
    val = 1/(1 + np.exp(-YY @ param))
    #TODO: implement if (all(dim(val)==c(1,1)))
    return val

def logit(x):
    r"""Logit function.
    
    It is the inverse of the
    standard logistic function, aka
    log-odds.
    
    :param x: the argument
    :type x: float
    :return: the logit of x
    :rtype: float
    """
    return np.log(x/(1-x))

def expit(x):
    r"""Expit function.
    
    It is the inverse of logit. Aka
    sigmoid or standard logistic.
    
    :param x: the argument
    :type x: float
    :return: the expit of x
    :rtype: float
    """
    return 1/(1+np.exp(-x))

def bitgamma(sample, m, W, gamma):
    r"""Shifted Binomial distribution with covariates.

    Return the shifted Binomial probabilities of ordinal responses where the feeling component 
    is explained by covariates via a logistic link.

    :param sample: array of ordinal responses
    :type sample: array
    :param m: number of ordinal categories
    :type m: int
    :param W: dataframe of covariates for explaining the feeling component;
        no column must be named ``0`` nor ``constant``
    :type W: pandas dataframe
    :param gamma: array :math:`\pmb \gamma` of parameters for the feeling component, whose length equals 
        ``W.columns.size+1`` to include an intercept term in the model (first entry)
    :type gamma: array of float
    :return: an array of the same length as ``sample``, where each entry is the shifted Binomial probability for
        the corresponding observation and feeling value.
    :rtype: array
    """
    ci = 1/logis(Y=W, param=gamma) - 1
    bg = kkk(sample=sample, m=m) * np.exp(
        (sample-1)*np.log(ci)-(m-1)*np.log(1+ci))
    return bg
    
def bitxi(m, sample, xi):
    r"""Shifted Binomial probabilities of ordinal responses

    Compute the shifted Binomial probabilities of ordinal responses.

    :param m: number of ordinal categories
    :type m: int
    :param sample: array of ordinal responses
    :type sample: array
    :param xi: feeling parameter :math:`\xi`
    :type xi: float
    :return: A vector of the same length as ``sample``, where each entry is the shifted Binomial probability 
        of the corresponding observation.
    :rtype: array
    """
    base = np.log(1-xi)-np.log(xi)
    cons = np.exp(m*np.log(xi)-np.log(1-xi))
    cons *= kkk(sample=sample, m=m)*np.exp(base*sample)
    return cons

def hadprod(Amat, xvett):
    r"""Hadamard product of a matrix with a vector

    Return the Hadamard product between the given matrix and vector: this operation corresponds 
    to multiply every row of the matrix by the corresponding element of the vector, and it is equivalent to the 
    standard matrix multiplication to the right with the diagonal matrix whose diagonal is the given vector. 
    It is possible only if the length of the vector equals the number of rows of the matrix.
    It is an auxiliary function needed for computing the variance-covariance matrix of the estimated model 
    with covariates.

    .. note:: if ``xvett`` is a row vector, reshapes it to column vector

    :param Amat: A generic matrix
    :type Amat: ndarray
    :param xvett: A generic vector
    :type xvett: array
    :return: the Hadamard product :math:`\pmb A \odot \pmb x`
    :rtype: ndarray
    """
    if isinstance(xvett, pd.Series):
        xvett = xvett.values
    if len(xvett.shape)==1:
        xvett = xvett.reshape(
            xvett.size, 1
        )
    dprod = Amat*xvett
    #ra = Amat.shape[0]
    #ca = Amat.shape[1]
    #dprod = np.zeros(shape=(ra, ca))
    #for i in range(ra):
    #    dprod[i,:] = Amat[i,:] * xvett[i]
    return dprod

def conf_ell(vcov, mux, muy, ci,
    ax, #showaxis=True, 
    color="b", label=True,
    alpha=.25):
    r"""Plot bivariate confidence ellipse of estimated
    parameters at level ``ci``:math:`=(1 - \alpha/2)`

    :param vcov: Variance-covariance matrix :math:`2 \times 2`
    :type vcov: ndarray
    :param mux: estimate of first parameter
    :type mux: float
    :param muy: estimate of second parameter
    :type muy: float
    :param ci: confidence level :math:`=(1 - \alpha/2)`
    :type ci: float
    :param ax: matplotlib axis
    :param color: color of confidence ellipse
    :type color: str
    :param label: whether to add a label of confidence level
    :type label: bool
    :param alpha: transparency of confidence ellipse
    :type alpha: float
    """
    nstd = np.sqrt(sps.chi2.isf(1-ci, df=2))
    #nstd = sps.norm().ppf((1-ci)/2)
    rho = vcov[0,1]/np.sqrt(vcov[0,0]*vcov[1,1])
    # beta1 = vcov[0,1] / vcov[0,0]
    radx = np.sqrt(1+rho)
    rady = np.sqrt(1-rho)
    ell = Ellipse(
        (0,0), 2*radx, 2*rady,
        color=color, alpha=alpha,
        label=f"CR {ci:.0%}" if label else None
    )
    scale_x = np.sqrt(vcov[0, 0]) * nstd
    scale_y = np.sqrt(vcov[1, 1]) * nstd
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mux, muy)
    ell.set_transform(transf + ax.transData)
    ax.add_patch(ell)
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
    """Load a saved object from file.

    It can used be used to load a ``CUBsample`` or a
    ``CUBres`` object, previously saved on a file.

    .. note:: see the Classes for details about these objects

    :param fname: filename
    :type fname: str
    :return: the loaded object, instance of ``CUBsample`` or ``CUBres``
    :rtype: object
    """
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    print(f"Object `{obj}` loaded from {fname}")
    return obj

def formula_parser(formula,
    model="cub"):
    r"""Parse a CUB class formula.

    Auxiliary function of ``cubmods.gem.from_formula()``.

    TODO: adapt formula to specific family with corresponding
        number of parameters
    
    TODO: documentation about formula

    TODO: add specific Exceptions for formula

    :param formula: the formula to be parsed
    :type formula: str
    :return: a tuple of the ordinal response column name and a list of all
        covariates' column names for each component
    :rtype: tuple
    """
    if '~' not in formula:
        raise Exception("ERR: ~ missing")
    # remove spaces
    formula = formula.replace(" ", "")
    # check formula
    reg = "^([a-zA-Z0-9_()]{1,})"
    rag = "([a-zA-Z0-9_()+]{1,})"
    #TODO: better formula regex?
    regex = f"{reg}~"
    comp = 2
    if model in ["cube", "cubsh"]:
        comp = 3
    elif model in ["cush", "ihg"]:
        comp = 1
    elif model=="cubsh2":
        comp = 4
    if formula.count("|") != comp-1:
        raise Exception(
        f"ERR: in {model} | must be {comp-1}")
    regex += "\|".join(np.repeat(rag,comp))
    regex += "$"
    #print(regex)
    if not re.match(regex, formula):
        raise Exception("ERR: wrong formula")
        #print("ERR: wrong formula")
        #return None
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
        if x == "1":
            XX.append([])
            continue
        x = x.split("+")
        XX.append(x)
    return y, XX

def unique(l):
    """Unique elements in a 3-dimensional list.

    Auxiliary function of ``.dummies2()``.

    :param l: the list to analyze
    :type l: list
    :return: the list of unique elements
    :rtype: list
    """
    a = []
    for i in l:
        if i is None:
            a.append(i)
            continue
        for j in i:
            a.append(j)
    u = list(set(a))
    return u

def dummies2(df, DD):
    r"""Create dummy variables from polychotomous variables.

    Auxiliary function of ``cubmods.gem.from_formula()``.
    A dummy variable is created for all polychotomous variables named
    ``C(<varname>)``.

    :param df: a DataFrame with all the covariates and the ordinal response
    :type df: DataFrame
    :param DD: the list of all covariates for each component
    :type DD: list
    :return: a tuple of the DataFrame with the dummy variables and the column names
    :rtype: tuple
    """
    # new covars
    XX = []
    # unique columns
    colnames = unique(DD)
    #print(colnames)
    # create dummy vars if any
    for c in colnames:
        if c is None:
            continue
        if c[:2]=="C(" and c[-1]==")":
            c = c[2:-1]
            # int to avoid floats
            if is_float_dtype(df[c]):
                df[c] = df[c].astype(int)
            df = pd.get_dummies(
                df, columns=[c],
                drop_first=True,
                prefix=f"C.{c}"
            )
    # define new covar names
    for D in DD:
        if D is None:
            XX.append(None)
            continue
        X = []
        for d in D:
            if d[:2]=="C(" and d[-1]==")":
                c = d[2:-1]
                # dummy names
                f = [f"C.{c}_" in i for i in df.columns]
                dums = df.columns[f]
                for dum in dums:
                    X.append(dum)
            else:
                X.append(d)
        XX.append(X)
                
    return df, XX

def _dummies(df, DD):
    """
    .. warn:: DEPRECATED
    """
    # new covars
    XX = []
    for j,D in enumerate(DD):
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
    """Exception: if m is not suitable for model.
    """
    def __init__(self, m, model):
        self.m = m
        self.model = model
        self.msg = f"Insufficient categories {self.m} for model {self.model}"
        super().__init__(self.msg)

class UnknownModelError(Exception):
    """Exception: if the requested family is unknown.
    """
    def __init__(self, model):
        self.model = model
        self.msg = f"Unknown model {self.model}"
        super().__init__(self.msg)

class NotImplementedModelError(Exception):
    """Exception: if the requested model is known but not
    yet implemented.
    """
    def __init__(self, model, formula):
        self.formula = formula
        self.model = model
        self.msg = f"Not implemented model {self.model} with formula {self.formula}"
        super().__init__(self.msg)

class NoShelterError(Exception):
    """Exception: if a shelter choice is needed but it hasn't been provided.
    """
    def __init__(self, model):
        self.model = model
        self.msg = f"Shelter choice (sh) needed for {self.model} model"
        super().__init__(self.msg)

#TODO: add in draw & mle in cubsh, cush
class ShelterGreaterThanM(Exception):
    """Exception: if the provided shelter choice is greater than :math:`m`.
    """
    def __init__(self, m, sh):
        self.m = m
        self.sh = sh
        self.msg = f"Shelter choice must be in [1,m], given sh={self.sh} with m={self.m}"
        super().__init__(self.msg)

#TODO: add in all draw
class ParameterOutOfBoundsError(Exception):
    """Exception: if the provided parameter value is out of bounds.
    """
    def __init__(self, param, value):
        self.param = param
        self.value = value
        self.msg = f"{self.value} is out of bounds for parameter {self.param}"
        super().__init__(self.msg)

#TODO: add in all draw
class InvalidSampleSizeError(Exception):
    """Exception: if the sample size is not strictly greater than zero.
    """
    def __init__(self, n):
        self.n = n
        self.msg = f"Sample size must be strictly > 0, given {self.n}"
        super().__init__(self.msg)

#########################################
# TEST TRIVARIATE CONFIDENCE ELLIPSPOID
# WITH BIVARIATE MARGINAL PROJECTIONS

#########################################

def get_minor(A, i, j):
    """Get a minor of a matrix.
    
    Auxiliary function of ``.plot_ellipsoid()``.

    .. note:: Solution by PaulDong

    :param A: a generic matrix
    :type A: ndarray
    :param i: row of the minor
    :type i: int
    :param j: column of the minor
    :type j: int
    :return: the minor of ``A``
    :rtype: ndarray
    """
    return np.delete(
        np.delete(A, i, axis=0), j, axis=1)

def conf_border(Sigma, mx, my, ax, conf=.95,
    plane="z", xyz0=(0,0,0)):
    """Plot the bivariate projection of a trivariate confidence ellipse
    on a plane.

    Auxiliary function of ``plot_ellipsoid()``.

    .. note:: Solution by https://gist.github.com/randolf-scholz.
    
    :param Sigma: bivariate variance-covariance matrix
    :type Sigma: ndarray
    :param mx: center of the ellipse on the :math:`x` axies
    :type mx: float
    :param my: center of the ellipse on the :math:`y` axies
    :type my: float
    :param ax: matpplotlib axis
    :param conf: confidence level of the trivariate ellipsoid.
    :type conf: float
    :param plane: plane for the projection; could be ``x``, ``y`` or ``z``
    :type plane: str
    :param xyz0: tuple of the bivariate ellipse position
    :type xyz0: tuple
    """
    #n = Sigma.shape[0]
    s = 1000
    # the 2d confidemce region, projection
    # of a 3d confidence region at ci%,
    # has got area = sqrt(ci^3)%
    r = np.sqrt(sps.chi2.isf(1-conf, df=2))
    #r = np.sqrt(sps.chi2.isf(
    #    1-np.cbrt(conf)**2, df=n))
    T = np.linspace(0, 2*np.pi, num=s)
    circle = r * np.vstack(
        [np.cos(T), np.sin(T)])
    x, y = sqrtm(Sigma) @ circle
    x += mx
    y += my
    if plane == "z":
        ax.plot(x,y,np.repeat(xyz0[2],s),"b")
        ax.plot(mx,my,xyz0[2],"ob")
    if plane == "y":
        ax.plot(x,np.repeat(xyz0[1],s),y,"b")
        ax.plot(mx,xyz0[1],my,"ob")
    if plane == "x":
        ax.plot(np.repeat(xyz0[0],s),
            x,y,"b",
            label=fr"CR {conf:.1%} $\in\mathbb{{R}}^2$")
        ax.plot(xyz0[0],mx,my,"ob")

def get_cov_ellipsoid(cov,
    mu=np.zeros((3)), ci=.95):
    r"""Return the 3d points representing the covariance matrix
    ``cov`` centred at ``mu``, at confidence level ``ci``:math:`=(1 - \alpha/2)`.

    Auxiliary function of ``.plot_ellipsoid()``.

    :param cov: Variance-covariance matrix :math:`3 \times 3`
    :type cov: ndarray
    :param mu: ellispoid center :math:`(x_0, y_0, z_0)`
    :type mu: array
    :param ci: confidence level :math:`=(1 - \alpha/2)`
    :type ci: float
    :return: a tuple of 3d points ``(X, Y, Z)``
    :rtype: tuple
    """
    assert cov.shape==(3,3)
    r = np.sqrt(sps.chi2.isf(1-ci, df=3))
    #nstd = sps.norm().ppf((1-ci)/2)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = r * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

def plot_ellipsoid(V, E, ax, zlabel,
    ci=.95, magnified=False):
    r"""Plot a trivariate confidence ellipsoid.

    :param V: Variance-covariance matrix
    :type V: ndarray
    :param E: Vector of estimated parameters
    :type E: array
    :param ax: matplotlib axis
    :param zlabel: label for :math:`z` axis
    :type zlabel: str
    :param ci: confidence level :math:`(1 - \alpha/2)`
    :type ci: float
    :param magnified: if ``False`` plots in the full parameter space
    :type magnified: bool
    """
    X,Y,Z = get_cov_ellipsoid(V, E, ci=ci)
    
    ax.scatter(*E, c='k', )
    ax.plot_wireframe(X,Y,Z, color='k',
        alpha=0.25,
        zorder=np.inf,
        label=fr"CR {ci:.0%} $\in\mathbb{{R}}^3$")
    #ax.plot_surface(X, Y, Z,
    #    edgecolor='k',
    #    lw=0.5,
    #    rstride=15, cstride=10,
    #                alpha=0.1)
    if not magnified:
        ax.set(
        zlim=[0,1], xlim=[0,1], ylim=[0,1])
    else:
        #ax.margins(.2)
        xl, yl, zl = equal3d(ax)

        ax.set(
            xlim=xl, ylim=yl, zlim=zl
        )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.plot(
        [E[0], E[0]],
        [E[1], E[1]],
        [E[2], zlim[0]],
        "r--"
    )
    ax.plot(
        [E[0], xlim[0]],
        [E[1], E[1]],
        [E[2], E[2]],
        "r--"
    )
    ax.plot(
        [E[0], E[0]],
        [E[1], ylim[1]],
        [E[2], E[2]],
        "r--"
    )
    #print(dir(ax.transData))
    #print(V.round(7))
    #print(E)
    minors = [2,   1,   0    ]
    planes = ["z", "y", "x"  ]
    #zs = [zlim[0], ylim[1], xlim[0]]
    for m, p in zip(minors, planes):
        minor = get_minor(V, m, m)
        #print(f"Plane: {p}")
        #print(minor)
        #if minor[0,1] != minor[1,0]:
        #    minor[[1,0],:] = minor[[0,1],:]
        mus = np.delete(E, m)
        #print(minor)
        #print(mus)
        ci2 = sps.chi2.cdf(
            sps.chi2.isf(1-ci, df=3),
            df=2
        )
        conf_border(minor, *mus, plane=p,
            ax=ax, conf=ci2,
            xyz0=(
                xlim[0], ylim[1], zlim[0]
            ))
    ax.set(
        xlim=xlim, ylim=ylim, zlim=zlim,
        #zlim=[0,1], xlim=[0,1], ylim=[0,1],
        xlabel=r"Uncertainty $(1-\pi)$",
        ylabel=r"Feeling $(1-\xi)$",
        zlabel=zlabel
    )
    ax.legend(loc="center right",
        bbox_to_anchor=(0,.5),
        frameon=0
    )

def equal3d(ax):
    r"""Equalize 3d axes.

    Auxiliary function of ``.plot_ellipsoid()``.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    # distances
    dxlim = np.diff(xlim)
    dylim = np.diff(ylim)
    dzlim = np.diff(zlim)
    # means
    mxlim = np.mean(xlim)
    mylim = np.mean(ylim)
    mzlim = np.mean(zlim)
    # max distance
    maxlim = np.max([dxlim,dylim,dzlim])
    # equal limits
    exlim = (mxlim-maxlim/2,mxlim+maxlim/2)
    eylim = (mylim-maxlim/2,mylim+maxlim/2)
    ezlim = (mzlim-maxlim/2,mzlim+maxlim/2)
    
    return exlim, eylim, ezlim
