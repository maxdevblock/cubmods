"""
CUB models in Python.
Module for MULTICUB and MULTICUBE.

Description:
============
    This module contains methods and classes
    for MULTICUB and MULTICUBE tool.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import multicub

    TODO: add MULTICUB example


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

import numpy as np
import matplotlib.pyplot as plt
from .gem import from_formula
from.general import (
    NotImplementedModelError,
    conf_ell
)

def pos_kwargs(pos):
    """
         1
       8   2
     7   @   3
       6   4
         5
    """
    if pos == 1:
        return dict(ha="center", va="bottom")
    if pos == 2:
        return dict(ha="left", va="bottom")
    if pos == 3:
        return dict(ha="left", va="center")
    if pos == 4:
        return dict(ha="left", va="top")
    if pos == 5:
        return dict(ha="center", va="top")
    if pos == 6:
        return dict(ha="right", va="top")
    if pos == 7:
        return dict(ha="right", va="center")
    if pos == 8:
        return dict(ha="right", va="bottom")
    # default if not allowed pos value
    return dict(ha="center", va="bottom")

def multi(ords, ms=None,
    model="cub",
    title=None,
    labels=None, shs=None,
    plot=True, print_res=False,
    pos=None, #position of phi/delta
    xlim=(0,1), ylim=(0,1),
    equal=True, #equal axes
    #conf ellipses params
    confell=True,
    alpha=.2, ci=.95,
    figsize=(7,7),
    ax=None):
    """
    ords: DataFrame
    ms:   list of m
    """
    allowed = ["cub", "cube"]
    if model not in allowed:
        raise NotImplementedModelError(
            model=model,
            formula="ord~0|0|0"
        )
        
    n = ords.columns.size
    if labels is not None:
        assert n == len(labels)
    if shs is not None:
        assert n == len(shs)
    if ms is None:
        ms = np.repeat(None, n)
    assert n == len(ms)
    
    ests = []
    for i in range(n):
        cname = ords.columns[i]
        sh = shs[i] if shs is not None else None
        #print(cname)
        est = from_formula(
            f"{cname}~0|0|0",
            model=model,
            df=ords,
            sh=sh,
            m=ms[i]
        )
        ests.append(est)
        if print_res:
            print(f"----> {cname} <----")
            print(est.summary())
    
    if plot:
        if title is None:
            title = f"MULTICUB. Model {model.upper()}"
            if shs is not None and model == "cub":
                title += "SH"
            if confell and model == "cub" and shs is None:
                title += f"    CR {ci:.0%}"
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize,
            )
        for i, est in enumerate(ests):
            pi = est.estimates[0]
            xi = est.estimates[1]
            cn = ords.columns[i]
            ax.plot(
                1-pi, 1-xi, "o",
                label=cn if labels is None else labels[i]
            )
            posi = pos_kwargs(1)
            if pos is not None:
                posi = pos_kwargs(pos[i])
            if model == "cube":
                phi = est.estimates[2]
                ax.text(1-pi, 1-xi,
                "\n"fr" $\phi={phi:.2f}$ ""\n",
                **posi, color=f"C{i}")
            if model == "cub" and shs is not None:
                delta = est.estimates[2]
                ax.text(1-pi, 1-xi,
                "\n"fr" $\delta={delta:.2f}$ ""\n",
                **posi, color=f"C{i}")
            if model == "cub" and shs is None and confell:
                conf_ell(vcov=est.varmat,
                    mux=1-pi, muy=1-xi,
                    ax=ax, label=False,
                    color=f"C{i}",
                    alpha=alpha, ci=ci)
        ax.set_title(title)
        ax.set_xlabel(r"Uncertainty $(1-\pi)$")
        ax.set_ylabel(r"Feeling $(1-\xi)$")
        ax.grid(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if xlim == (0,1):
            ax.set_xticks(np.arange(0, 1.1, .1))
        if ylim == (0,1):
            ax.set_yticks(np.arange(0, 1.1, .1))
        if equal:
            ax.set_aspect("equal")
        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(4)
        # increase tick width
            ax.tick_params(width=4)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        return ax
