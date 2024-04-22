"""
CUB models in Python.
Module for GEM (Generalized Mixtures).

Description:
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

References:
    * TODO: add references

List of TODOs:
    * TODO: implement best shelter search

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""
# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements
# pylint: disable=dangerous-default-value, too-many-branches
import warnings
import numpy as np
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

def from_formula(
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
    ordinal, covars = formula_parser(formula)
    #print(ordinal, covars)
    # all rows with at least a NaN will be dropped
    df.dropna(inplace=True)
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
        if covars[2] is not None:
            print("ERR: only Y and W are covariates for cub model")
            return None
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
        if covars[1] is not None or covars[2] is not None:
            print("ERR: only X are covariates for cush model")
            return None
        if sh is None:
            if sh is None:
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
        raise UnknownModelError(model=model)
        #print(f"No implemented model {model} with formula {formula}")
        #return None

    fit = mod.mle(
            **pars,
            **options,
            gen_pars=gen_pars
        )
    return fit
