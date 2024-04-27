"""
CUB models in Python.
Module for General functions.

Description:
============
    This module contains methods and classes
    for general functions.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    add example


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
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements
# pylint: disable=anomalous-backslash-in-string
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

def lsat(f, n):
    # loglik of saturated model
    logliksat = -(n*np.log(n)) + np.sum((f[f!=0])*np.log(f[f!=0]))
    return logliksat

#TODO: add loglikbin to all models and smry ???
def lbin(sample, m, f):
    avg = sample.mean()
    xi = (m-avg)/(m-1)
    R = choices(m)
    p = binom(m-1, R-1) * (1-xi)**(R-1) * xi**(m-R)
    l = np.sum(f*np.log(p))
    return l

#TODO: is lsatcov useful?
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

def conf_ell(vcov, mux, muy, ci,
    ax, #showaxis=True, 
    color="b", label=True,
    alpha=.25):
    """
    plot confidence ellipse of estimated
    CUB parameters of ci% on ax
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
    """
    Load a saved object from file
    """
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    print(f"Object `{obj}` loaded from {fname}")
    return obj

def formula_parser(formula):
    if '~' not in formula:
        raise Exception("ERR: ~ missing")
    if formula.count("|") != 2:
        raise Exception("ERR: | must be 2")
    regex = "^([a-zA-Z0-9_()]{1,})~"
    regex += "([a-zA-Z0-9_+()]{1,})\|"
    regex += "([a-zA-Z0-9_+()]{1,})\|"
    regex += "([a-zA-Z0-9_+()]{1,})$"
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
        x = x.split("+")
        XX.append(x)
    return y, XX

def unique(l):
    """
    unique column names in
    covar 3d list
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
    """
    
    """
    # new covars
    XX = []
    # unique columns
    colnames = unique(DD)
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

def dummies(df, DD):
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
        self.msg = f"Not implemented model {self.model} with formula {self.formula}"
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
class ParameterOutOfBoundsError(Exception):
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

#########################################
# TEST TRIVARIATE CONFIDENCE ELLIPSPOID
# WITH BIVARIATE MARGINAL PROJECTIONS

#########################################

def get_minor(A, i, j):
    """Solution by PaulDong"""
    return np.delete(
        np.delete(A, i, axis=0), j, axis=1)

def conf_border(Sigma, mx, my, ax, conf=.95,
    plane="z", xyz0=(0,0,0)):
    """Solution by
    https://gist.github.com/randolf-scholz"""
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
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
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
    """
    3d confidence ellipsoid
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
