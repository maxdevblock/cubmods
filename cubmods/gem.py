# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace, dangerous-default-value, too-many-branches
"""
CUB models in Python.
Module for GEM (Generalized Mixtures).

Description:
============
    This module contains methods and classes
    for GEM maximum likelihood estimation.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import gem

    samp = pd.read_csv("observed.csv")
    fit = gem.from_formula(
        formula="ordinal~0|0|0",
        model="cub",
        m=7
    )
    print(fit.summary())
    fit.plot()
    plt.show()


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
  TODO: implement best shelter search

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""

import warnings
import numpy as np
import pandas as pd
from . import (
    cub, cub_0w, cub_y0, cub_yw,
    cube, cube_0w0, cube_ywz,
    cubsh, cubsh_ywx,
    cush, cush_x,
    cush2, cush2_x0, cush2_xx,
    ihg, ihg_v
    )
from .general import (
    formula_parser, dummies2,
    UnknownModelError,
    NotImplementedModelError,
    NoShelterError
)

def estimate(
    formula,      # the formula to apply
    df,           # DataFrame of sample and covariates
    m=None,       # if None takes max(sample)
    model="cub",  # "cub", "cube", "cush", "cush2"
    sh=None,      # used for cubsh and cush only
    gen_pars=None,# dict of known generating params
    options={}    # "maxiter" and/or "tol"
    ):
    """
    Takes a DataFrame as input and calls MLE
    based upon given model and formula.
    """
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning
    )
    modname = model
    if model == "cub" and sh is not None:
        modname = "cubsh"
    ordinal, covars = formula_parser(formula,
        model=modname)
    #print(ordinal, covars)
    # all rows with at least a NaN will be dropped
    dfi_tot = df.index.size
    df = df.dropna().copy(deep=True)
    dfi_nona = df.index.size
    if dfi_tot != dfi_nona:
        warnings.warn(f"{dfi_tot-dfi_nona} NaNs detected and removed.")
    sample = df[ordinal]
    n = sample.size
    df, covars = dummies2(df=df, DD=covars)
    #TODO: other warnings?
    if n < 200:
        warnings.warn("Sample size less than 200")
    if m is None:
        warnings.warn("No m given, max(ordinal) has been taken")
        m = np.max(sample)

    if model=="cub" and sh is None:
        Y = covars[0] #covariates for pi
        W = covars[1] #covariates for xi
        # R~Y|W|$
#        if covars[2] is not None:
#            print("ERR: only Y and W are covariates for cub model")
#            return None
        # R~0|0|0
        if Y is None and W is None:
            #TODO: if m <=
            mod = cub
            pars = {"sample":sample, "m":m}
        # R~Y|W|0
        elif Y is not None and W is not None:
            #TODO: if m <=
            mod = cub_yw
            pars = {"sample":sample, "m":m, "Y":df[Y], "W":df[W]}
        # R~0|W|0
        elif Y is None and W is not None:
            #TODO: if m <=
            mod = cub_0w
            pars = {"sample":sample, "m":m, "W":df[W]}
        # R~Y|0|0
        elif Y is not None and W is None:
            #TODO: if m <=
            mod = cub_y0
            pars = {"sample":sample, "m":m, "Y":df[Y]}
    elif model=="cube":
        Y = covars[0] #covariates for pi
        W = covars[1] #covariates for xi
        Z = covars[2] #covariates for phi
        # R~0|0|0
        if Y is None and W is None and Z is None:
            #TODO: if m <=
            mod = cube
            pars = {"sample":sample, "m":m}
        # R~0|W|0
        elif Y is None and W is not None and Z is None:
            #TODO: if m <=
            mod = cube_0w0
            pars = {"sample":sample, "m":m, "W":df[W]}
        # R~Y|W|Z
        elif Y is not None and W is not None and Z is not None:
            #TODO: if m <=
            mod = cube_ywz
            pars = {"sample":sample, "m":m, "Y":df[Y], "W":df[W], "Z":df[Z]}
        else:
            raise NotImplementedModelError(model=model, formula=formula)
            #print(f"ERR(cube): no implemented model {model} with formula {formula}")
            #return None
    elif model=="cub" and sh is not None:
        Y = covars[0] #covariates for pi
        W = covars[1] #covariates for xi
        X = covars[2] #covariates for delta
        if not sh:
            print("WARN: searching for best shelter choice")
            #TODO: implement shelter choice search
        else:
            # R~0|0|0
            if Y is None and W is None and X is None:
                #TODO: if m <=
                mod = cubsh
                pars = {"sample":sample, "m":m, "sh":sh}
            # R~Y|W|X
            elif Y is not None and W is not None and X is not None:
                #TODO: if m <=
                mod = cubsh_ywx
                pars = {"sample":sample, "m":m, "sh":sh, "Y":df[Y], "W":df[W], "X":df[X]}
            else:
                raise NotImplementedModelError(model=model, formula=formula)
                #print(f"ERR(cubsh): no implemented model {model}sh with formula {formula}")
                #return None
    elif model=="cush":
        X = covars[0] #covariates for delta
        
        if sh is None:
            #if sh is None:
            raise NoShelterError(model=model)
        #TODO: if sh=0 search for the best shelter choice
        elif not sh:
            print("WARN: searching for best shelter choice")
            #TODO: implement shelter choice search
        else:
            if X is None:
                #TODO: if m <=
                mod = cush
                pars = {"sample":sample, "m":m, "sh":sh}
            elif X is not None:
                #TODO: if m <=
                mod = cush_x
                pars = {"sample":sample, "m":m, "sh":sh, "X":df[X]}
            else:
                raise NotImplementedModelError(model=model, formula=formula)
                #print(f"ERR: no implemented model {model} with formula {formula}")
                #return None
    elif model == "cush2":
        X1 = covars[0]
        X2 = covars[1]
        sh1 = sh[0]
        sh2 = sh[1]
        if X1 is None and X2 is None:
            mod = cush2
            pars = {"sample":sample,
                "m":m, "c1":sh1, "c2":sh2}
        elif X1 is not None and X2 is not None:
            mod = cush2_xx
            pars = {"sample":sample,
                "m":m, "sh1":sh1, "sh2":sh2,
                "X1":df[X1], "X2":df[X2]}
        elif X1 is not None and X2 is None:
            mod = cush2_x0
            pars = {"sample":sample,
                "m":m, "sh1":sh1, "sh2":sh2,
                "X1":df[X1]}
        else:
            raise NotImplementedModelError(model=model, formula=formula)
            #print(f"ERR: no implemented model {model} with formula {formula}")
            #return None
    elif model == "ihg":
        V = covars[0] # covariates for theta
        if V is None:
            mod = ihg
            pars = {"sample":sample, "m":m,}
        else:
            mod = ihg_v
            pars = {"sample":sample, "m":m, "V":df[V]}
    else:
        raise UnknownModelError(
        model=f"{model}"
            + f"with formula {formula}"
        )

    fit = mod.mle(
            **pars,
            **options,
            gen_pars=gen_pars,
            df=df, formula=formula
        )
    return fit

def draw(formula, df=None,
    m=7, model="cub", n=500,
    sh=None, seed=None,
    **params
    ):
    modname = model
    if model == "cub" and sh is not None:
        modname = "cubsh"
    ordinal, covars = formula_parser(formula,
        model=modname)
    if df is None:
        df = pd.DataFrame(
            index=np.arange(n))
    orig_df = df.copy(deep=True)
    #print(ordinal, covars)
    # all rows with at least a NaN will be dropped
    dfi_tot = df.index.size
    df = df.dropna().copy(deep=True)
    dfi_nona = df.index.size
    if dfi_tot != dfi_nona:
        warnings.warn(f"{dfi_tot-dfi_nona} NaNs detected and removed.")
    df, covars = dummies2(df=df, DD=covars)
    if model=="cub" and sh is None:
        Y = covars[0]
        W = covars[1]
        if Y is None and W is None:
            mod = cub
            params.update(dict(
            seed=seed, n=n, m=m
            ))
        if Y is None and W is not None:
            mod = cub_0w
            params.update(dict(
            seed=seed, W=df[W]
            ))
        if Y is not None and W is None:
            mod = cub_y0
            params.update(dict(
            seed=seed, Y=df[Y]
            ))
        if Y is not None and W is not None:
            mod = cub_yw
            params.update(dict(
            seed=seed, Y=df[Y], W=df[W]
            ))
    elif model=="cub" and sh is not None:
        Y = covars[0]
        W = covars[1]
        X = covars[2]
        if Y is None and W is None and X is None:
            mod = cubsh
            params.update(dict(
            seed=seed, sh=sh, n=n
            ))
        if Y is not None and W is not None and X is not None:
            mod = cubsh_ywx
            params.update(dict(
            seed=seed, sh=sh, m=m,
            Y=df[Y], W=df[W], X=df[X]
            ))
    elif model=="cube":
        Y = covars[0]
        W = covars[1]
        Z = covars[2]
        if Y is None and W is None and Z is None:
            mod = cube
            params.update(dict(
            seed=seed, n=n
            ))
        if Y is None and W is not None and Z is None:
            mod = cube_0w0
            params.update(dict(
            seed=seed, W=df[W]
            ))
        if Y is not None and W is not None and Z is not None:
            mod = cube_ywz
            params.update(dict(
            seed=seed, Y=df[Y], W=df[W],
            Z=df[Z]
            ))
    elif model=="cush":
        X = covars[0]
        if X is None:
            mod = cush
            params.update(dict(
            seed=seed, sh=sh, n=n
            ))
        if X is not None:
            mod = cush_x
            params.update(dict(
            seed=seed, sh=sh, X=df[X]
            ))
    elif model=="cush2":
        X1 = covars[0]
        X2 = covars[1]
        if X1 is None and X2 is None:
            mod = cush2
            params.update(dict(
            seed=seed, sh1=sh[0], sh2=sh[1],
            n=n
            ))
        if X1 is not None and X2 is None:
            mod = cush2_x0
            params.update(dict(
            seed=seed, sh1=sh[0], sh2=sh[1],
            X1=df[X1]
            ))
        if X1 is not None and X2 is not None:
            mod = cush2_xx
            params.update(dict(
            seed=seed, sh1=sh[0], sh2=sh[1],
            X1=df[X1], X2=df[X2]
            ))
    elif model=="ihg":
        V = covars[0]
        if V is None:
            mod = ihg
            params.update(dict(
            seed=seed, n=n
            ))
        if V is not None:
            mod = ihg_v
            params.update(dict(
            seed=seed, V=df[V]
            ))
    else:
        raise UnknownModelError(
        model=f"{model}"
            + f"with formula {formula}"
        )

    params.update(dict(
        df=df, formula=formula,
        m=m, orig_df=orig_df
    ))
    #print(params)
    return mod.draw(**params)
    
