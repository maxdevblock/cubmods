import numpy as np
import matplotlib.pyplot as plt
from .gem import from_formula
from.general import NotImplementedModelError

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

def multi(ords, ms,
    model="cub",
    title=None,
    labels=None, shs=None,
    plot=True, print_res=False,
    pos=None, #position of phi/delta
    figsize=(7,7)):
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
    assert n == len(ms)
    if labels is not None:
        assert n == len(labels)
    if shs is not None:
        assert n == len(shs)
    
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
        ax.set_title(title)
        ax.set_xlabel(r"Uncertainty $(1-\pi)$")
        ax.set_ylabel(r"Preference $(1-\xi)$")
        ax.grid(True)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        ax.set_xticks(np.arange(0, 1.1, .1))
        ax.set_yticks(np.arange(0, 1.1, .1))
        ax.set_aspect("equal")
        # change all spines
        for axis in ['left','bottom']:
            ax.spines[axis].set_linewidth(4)
        # increase tick width
            ax.tick_params(width=4)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        return fig, ax
