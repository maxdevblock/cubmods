# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, invalid-name, too-many-arguments, too-many-locals, too-many-statements, trailing-whitespace
"""
.. _multicub-module:

CUB models in Python.
Module for MULTICUB and MULTICUBE.

Description:
============
    This module contains methods and classes
    for MULTICUB and MULTICUBE tool.
    It is based upon the works of Domenico
    Piccolo et Al. and the ``CUB`` package in R.

Manual, Examples and References:
===============================
    - `Models manual <manual.html#multicub-manual>`__

List of TODOs:
==============
  - ...

Credits
==============
    :Author:      Massimo Pierini

    :Date:        2023-24
    :Credits:     Domenico Piccolo, Rosaria Simone
    :Contacts:    cub@maxpierini.it

Classes and Functions
=====================
"""

import numpy as np
import matplotlib.pyplot as plt
from .gem import estimate
from.general import (
    NotImplementedModelError,
    conf_ell
)

def pos_kwargs(pos):
    r"""Position of the :math:`\delta` or :math:`\phi` estimated values

    .. code-block:: none

            1
          8   2
        7   @   3
          6   4
            5

    :param pos: position (1..8)
    :type pos: int
    :return: a dictionary for ``matplotlib``
    :rtype: dict
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
    r"""Joint plot of estimated CUB models in the parameter space

    Return a plot of estimated CUB models represented as points in the parameter space.

    :param ords: list of arrays of observed ordinal responses
    :type ords: list
    :param model: model; defaults to ``cub``; options ``cube``
    :type model: str
    :param title: title of the plot
    :type title: str
    :param labels: labels of the points
    :type labels: list
    :param shs: shelter effect(s); can be an *int* if the same shelter
        effect is valid for all samples or a *list* to specify different
        shelter choices
    :type shs: int or list
    :param plot: if ``True`` (default) plot the results;
    :type plot: bool
    :param print_res: if ``True`` print the results; defaults to ``False``
    :type print_res: bool
    :param pos: position of the :math:`\delta` or :math:`\phi` estimated values
    :type pos: list
    :param xlim: x-axis limits
    :type xlim: tuple
    :param ylim: y-axis limits
    :type ylim: tuple
    :param equal: if the plot must have equal aspect; defaults to ``True``
    :type equal: bool
    :param alpha: confidence ellipse transparency
    :type alpha: float
    :param confell: if ``True`` (default) plot confidence ellipse (for CUB model only)
    :type confell: bool
    :param ci: level :math:`(1-\alpha/2)` for the confidence ellipse
    :type ci: float
    :param figsize: tuple of ``(length, height)`` for the figure (useful only if ``ax`` is not None)
    :type figsize: tuple of float
    :param ax: matplotlib axis, if None a new figure will be created, defaults to None
    :type ax: matplolib ax, optional
    :return: ``ax``
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
    if isinstance(shs, int):
        shs = np.repeat(shs, n)
    if shs is not None:
        assert n == len(shs)
    if ms is None:
        ms = np.repeat(None, n)
    if isinstance(ms, int):
        ms = np.repeat(ms, n)
    assert n == len(ms)
    
    ests = []
    for i in range(n):
        cname = ords.columns[i]
        sh = shs[i] if shs is not None else None
        comps = "0 | 0 | 0"
        if model == "cub" and sh is None:
            comps = "0 | 0"
        #print(cname)
        est = estimate(
            formula=f"{cname}~{comps}",
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
            title = f"MULTICUB. "
            modname = model
            if shs is not None and model == "cub":
                modname += "SH"
            title += f"{modname.upper()} models."
            if confell and model == "cub" and shs is None:
                title += f" {ci:.0%} confidence regions."
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize,
            )
        else:
            fig = None
        
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
            if model == "cub" and sh is not None:
                delta = est.estimates[-1]
                ax.text(1-pi, 1-xi,
                "\n"fr" $\delta={delta:.2f}$ ""\n",
                **posi, color=f"C{i}")
            if model == "cub" and sh is None and confell:
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
        return fig, ax
