# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace, dangerous-default-value, too-many-branches
"""
.. _gem-module:

CUB models in Python.
Module for GEM (Generalized Mixtures).

Description:
============
    This module contains methods and classes
    for GEM maximum likelihood estimation
    and sample drawing.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Manual and Examples
==========================
    - `Models manual <manual.html#gem-manual>`__

References:
===========
    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

  
List of TODOs:
==============
  - TODO: implement best shelter search

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
    model="cub",  # "cub", "cube", "cush"
    sh=None,      # used for cubsh and cush only
    gen_pars=None,# dict of known generating params
    options={}    # "maxiter" and/or "tol"
    ):
    r"""Main function to estimate and validate GEneralized Mixture models.

    :param formula: a formula used to estimate the model's parameters, see
        Manual for details
    :type formula: str
    :param df: the DataFrame with observed ordinal sample and covariates (if any)
    :type df: DataFrame
    :param m: number of ordinal categories
    :type m: int
    :param model: the model family; default to ``"cub"``; options ``"cube"`` and ``"cush"``
    :type model: str
    :param sh: category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param gen_pars: dictionary of hypothesized parameters, defaults to None
    :type gen_pars: dictionary, optional
    :param options: a dictionary of extra options ``maxiter`` and ``tol``; see the reference
        guide for details
    :type options: dict
    :return: an instance of the Base Class ``CUBres`` extended by the family module;
        see each module for details
    :rtype: obj
    """
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning
    )
    modname = model
    if model == "cub" and sh is not None:
        modname = "cubsh"
    elif model == "cush" and isinstance(sh, int):
        modname = "cush"
    elif model == "cush" and isinstance(sh, (list, np.ndarray)):
        modname = "cush2"
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
            raise NotImplementedModelError("Searching for best shelter choice not implemented yet.",
                                           formula)
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
    elif modname == "cush":
        X = covars[0] #covariates for delta
        
        if sh is None:
            #if sh is None:
            raise NoShelterError(model=model)
        #TODO: if sh=0 search for the best shelter choice
        elif not sh:
            raise NotImplementedModelError("Searching for best shelter choice not implemented yet.",
                                           formula)
            #TODO: implement shelter choice search
        #elif isinstance(sh, float):
        #TODO: implement cush2 from cush if sh is arraylike of size 2
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
    elif modname == "cush2":
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
    r"""Main function to draw a sample from GEneralized Mixture models.

    :param formula: a formula used to draw the sample, see
        Manual for details
    :type formula: str
    :param df: the DataFrame with covariates (if any)
    :type df: DataFrame
    :param m: number of ordinal categories
    :type m: int
    :param model: the model family; default to ``"cub"``; options ``"cube"`` and ``"cush"``
    :type model: str
    :param sh: category corresponding to the shelter choice :math:`[1,m]`
    :type sh: int
    :param n: number of ordinal responses; it is only effective if the model
        is without covariates
    :type n: int
    :param gen_pars: dictionary of hypothesized parameters, defaults to None
    :type gen_pars: dictionary, optional
    :param options: a dictionary of extra options ``maxiter`` and ``tol``; see the reference
        guide for details
    :type options: dict
    :param seed: the `seed` to ensure reproducibility, defaults to None
    :type seed: int, optional
    :return: an instance of ``CUBsample`` containing ordinal responses drawn from the specified model
    :rtype: obj
    """
    modname = model
    if model == "cub" and sh is not None:
        modname = "cubsh"
    elif model == "cush" and isinstance(sh, int):
        modname = "cush"
    elif model == "cush" and isinstance(sh, (list, np.ndarray)):
        modname = "cush2"
    _, covars = formula_parser(formula,
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
    elif modname=="cush":
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
    elif modname=="cush2":
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
        formula=formula,
        m=m, df=orig_df
    ))
    #print(params)
    return mod.draw(**params)
    
