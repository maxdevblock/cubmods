r"""
CUB models in Python.
Module for summary tools.

Description:
============
    This module contains methods and classes
    for summary tools.
    It is based upon the works of Domenico
    Piccolo et Al. and CUB package in R.

Example:
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import cub

    samp = pd.read_csv("ordinal.csv")
    fit = cub.mle(samp.rv, m=7)
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
  *TODO: Correlation su, --- invece di === come nella tesi
  *TODO: loglik invece di loglike
  *TODO: Corr(\pi,\xi) invece di Corr(pi, xi) nei plot
  *TODO: Feeling invece di Preference nei grafici
  *TODO: barre invece di pallini (aggiungere opzione)
  *TODO: controllare NAs (e dirlo)
  TODO: risultati inferenziali come DataFrame nel Manuale e negli esempi
  TODO: User Manual
  TODO: Refernce Guide
  TODO: bounds opzionali in CUBE mle (anche CUBSH?)
  TODO: 2 decimali nei 3d plot?
  *TODO: stessa unità di misura nei 3d plot magnified?
  *TODO: titolo con covariate AVERAGE ESTIMATED PROBABILITY
  TODO: dissim in multicub plot (aggiungere opzione)
  TODO: grandezza punti phi in multicube
  *TODO: stessi decimali nel summary

@Author:      Massimo Pierini
@Institution: Universitas Mercatorum
@Affiliation: Graduand in Statistics & Big Data (L41)
@Date:        2023-24
@Credit:      Domenico Piccolo, Rosaria Simone
@Contacts:    cub@maxpierini.it
"""

#import datetime as dt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .general import (
    choices, freq
)

def as_txt(
    model, m, n, sh,
    maxiter, niter, tol, p,
    e_types, est_names,
    estimates, stderrs, wald, pval,
    loglike, muloglik, loglikuni,
    logliksat, logliksatcov, dev,
    diss, AIC, BIC, rho,
    seconds, time_exe,
    # unused for Class compatibility
    **kwargs,
    #theoric, sample, f, varmat,
    #V, W, X, Y, Z, gen_pars
    ):
    """
        DEPRECATED FUNCTION.
    """
    par_names = np.asarray(est_names)
    par_types = np.asarray(e_types)
    lparnames = len(max(par_names, key=len))
    pars = np.asarray(estimates)
    pars = np.round(pars, 3)
    #print(pars)
    #lpars = len(max(pars.astype(str), key=len))
    space1 = lparnames+2
    #print(space1)
    ses = np.asarray(stderrs)
    ses = np.round(ses, 4)
    lses = len(max(ses.astype(str), key=len))
    space2 = max([6, lses])+2-6
    walds = np.asarray(wald)
    walds = np.round(walds, 3)
    lwalds = len(max(walds.astype(str), key=len))
    space3 = max([4, lwalds])+2-4
    pvals = np.asarray(pval)
    pvals = np.round(pvals, 4)
    #lpvals = len(max(pvals.astype(str), key=len))
    space4 = 2
    
    sep = "=======================================================================\n"
    sup = "-----------------------------------------------------------------------\n"
    est = f"{' '*space1}Estimates{' '*space2}StdErr{' '*space3}Wald{' '*space4}p-value\n"
    
    smry = sep
    smry += f"=====>>> {model.upper()} model <<<===== ML-estimates\n"
    smry += sep
    smry += f"m={m}  "
    if sh is not None:
        smry += f"Shelter={sh}  "
    smry += f"Size={n}  "
    if niter is not None:
        smry += f"Iterations={niter}  "
    if maxiter is not None:
        smry += f"Maxiter={maxiter}  "
    if tol is not None:
        smry += f"Tol={tol:.0E}"
    smry += "\n"
    for i in range(p):
        if par_types[i] is not None:
            smry += sup
            smry += f"{par_types[i]}\n"
            smry += est
        spaceA = (space1+9)-len(par_names[i])-(len(f"{pars[i]:+}"))
        spaceB = space2+6-len(str(ses[i]))
        spaceC = space3+4-len(str(walds[i]))
        spaceD = space4+7-6
        #print(f"`{str(pars[i])}`")
        smry += f"{par_names[i]}{' '*spaceA}{pars[i]:+}{' '*spaceB}{ses[i]}{' '*spaceC}{walds[i]}{' '*spaceD}{pvals[i]:.4f}"
        smry += "\n"
    if rho is not None:
        smry += sup
        smry += f"Correlation   = {rho:.4f}\n"
    smry += sep
    if diss is not None:
        smry += f"Dissimilarity = {diss:.4f}\n"
    ls = "  "
    l_ = None
    c_ = None
    if (
        kwargs["V"] is not None or
        kwargs["W"] is not None or
        kwargs["X"] is not None or
        kwargs["Y"] is not None or
        kwargs["Z"] is not None
        ):
        ls = "* "
        if logliksat is not None:
            l_ = "* Saturated model without covariates\n"
    if logliksatcov is not None:
        c_ = "^ not valid for continuous covariates\n"
        smry += f"Logl(satcov)^ = {logliksatcov:.3f}\n"
    warn = ""
    if logliksat is not None:
        warn = " (!)" if logliksat<loglike else ""
        smry += f"Loglik(sat) {ls}= {logliksat:.3f}{warn}\n"
    smry += f"Loglik(MOD)   = {loglike:.3f}\n"
    smry += f"Loglik(uni)   = {loglikuni:.3f}\n"
    smry += f"Mean-loglik   = {muloglik:.3f}\n"
    if dev is not None:
        smry += f"Deviance{ls}    = {dev:.3f}{warn}\n"
    if c_ is not None:
        smry += c_
    if l_ is not None:
        smry += l_
    smry += sup
    smry += f"AIC = {AIC:.2f}\n"
    smry += f"BIC = {BIC:.2f}\n"
    smry += sep
    smry += f"Elapsed time={seconds:.5f} seconds =====>>> {time_exe:%c}\n"
    smry += sep[:-1]
    return smry

class CUBres(object):
    def __init__(
        self,
        model, m, n,
        sample, f, theoric, diss,
        est_names, estimates, e_types,
        varmat, stderrs, pval, wald,
        loglike, muloglik,
        loglikuni,
        AIC, BIC,
        seconds, time_exe,
        # optional parameters
        logliksat=None, dev=None,
        logliksatcov=None,
        niter=None, maxiter=None, tol=None,
        Y=None, W=None, X=None,
        V=None, Z=None, sh=None,
        rho=None, gen_pars=None,
    ):
        self.model = model
        self.m = m
        self.n = n
        self.sh = sh
        self.niter = niter
        self.maxiter = maxiter
        self.tol = tol
        self.theoric = theoric
        self.estimates = np.array(estimates)
        self.est_names = np.array(est_names)
        self.e_types = np.array(e_types)
        self.stderrs = np.array(stderrs)
        self.pval = np.array(pval)
        self.wald = np.array(wald)
        self.loglike = loglike
        self.muloglik = muloglik
        self.loglikuni = loglikuni
        self.logliksat = logliksat
        self.logliksatcov = logliksatcov
        self.dev = dev
        self.AIC = AIC
        self.BIC = BIC
        self.seconds = seconds
        self.time_exe = time_exe
        self.rho = rho
        self.sample = sample
        self.f = f
        self.varmat = varmat
        self.V = V
        self.W = W
        self.X = X
        self.Y = Y
        self.Z = Z
        self.diss = diss
        self.gen_pars = gen_pars
        # number of parameters
        self.p = self.estimates.size

    def __str__(self):
        pars = ""
        for i in range(self.p):
            pars += f"{self.est_names[i]}={self.estimates[i]:.3f}"
            if i < (self.p-1):
                pars += "; "
        return f"CUBres({self.model}; {pars})"

    def as_txt(self):
        par_names = np.asarray(self.est_names)
        par_types = np.asarray(self.e_types)
        lparnames = len(max(self.est_names, key=len))
        pars = np.asarray(self.estimates)
        parsT = []
        for i in range(pars.size):
            parsT.append(f"{pars[i]:.3f}")
        #print(pars)
        #lpars = len(max(pars.astype(str), key=len))
        space1 = lparnames+2
        #print(space1)
        ses = np.asarray(self.stderrs)
        sesT = []
        for i in range(ses.size):
            sesT.append(f"{ses[i]:.4f}")
        lses = len(max(sesT, key=len))
        space2 = max([6, lses])+2-6
        walds = np.asarray(self.wald)
        waldsT = []
        for i in range(walds.size):
            waldsT.append(f"{walds[i]:.3f}")
        lwalds = len(max(waldsT, key=len))
        space3 = max([4, lwalds])+2-4
        pvals = np.asarray(self.pval)
        pvalsT = []
        for i in range(pvals.size):
            pvalsT.append(f"{pvals[i]:.4f}")
        #lpvals = len(max(pvals.astype(str), key=len))
        space4 = 2
        
        sep = "=======================================================================\n"
        sup = "-----------------------------------------------------------------------\n"
        est = f"{' '*space1}Estimates{' '*space2}StdErr{' '*space3}Wald{' '*space4}p-value\n"
        
        smry = sep
        smry += f"=====>>> {self.model.upper()} model <<<===== ML-estimates\n"
        smry += sep
        smry += f"m={self.m}  "
        if self.sh is not None:
            smry += f"Shelter={self.sh}  "
        smry += f"Size={self.n}  "
        if self.niter is not None:
            smry += f"Iterations={self.niter}  "
        if self.maxiter is not None:
            smry += f"Maxiter={self.maxiter}  "
        if self.tol is not None:
            smry += f"Tol={self.tol:.0E}"
        smry += "\n"
        for i in range(self.p):
            if par_types[i] is not None:
                smry += sup
                smry += f"{par_types[i]}\n"
                smry += est
            spaceA = (space1+9)-len(par_names[i])-(len(parsT[i]))
            spaceB = space2+6-len(sesT[i])
            spaceC = space3+4-len(waldsT[i])
            spaceD = space4+7-6
            #print(f"`{str(pars[i])}`")
            smry += f"{par_names[i]}{' '*spaceA}{parsT[i]}{' '*spaceB}{sesT[i]}{' '*spaceC}{waldsT[i]}{' '*spaceD}{pvalsT[i]}"
            smry += "\n"
        if self.rho is not None:
            smry += sup
            smry += f"Correlation   = {self.rho:.4f}\n"
        smry += sep
        if self.diss is not None:
            smry += f"Dissimilarity = {self.diss:.4f}\n"
        ls = "  "
        l_ = None
        c_ = None
        if (
            self.V is not None or
            self.W is not None or
            self.X is not None or
            self.Y is not None or
            self.Z is not None
            ):
            ls = "* "
            if self.logliksat is not None:
                l_ = "* Saturated model without covariates\n"
        if self.logliksatcov is not None:
            c_ = "^ not valid for continuous covariates\n"
            smry += f"Logl(satcov)^ = {self.logliksatcov:.3f}\n"
        warn = ""
        if self.logliksat is not None:
            warn = " (!)" if self.logliksat<self.loglike else ""
            smry += f"Loglik(sat) {ls}= {self.logliksat:.3f}{warn}\n"
        smry += f"Loglik(MOD)   = {self.loglike:.3f}\n"
        smry += f"Loglik(uni)   = {self.loglikuni:.3f}\n"
        smry += f"Mean-loglik   = {self.muloglik:.3f}\n"
        if self.dev is not None:
            smry += f"Deviance{ls}    = {self.dev:.3f}{warn}\n"
        if c_ is not None:
            smry += c_
        if l_ is not None:
            smry += l_
        smry += sup
        smry += f"AIC = {self.AIC:.2f}\n"
        smry += f"BIC = {self.BIC:.2f}\n"
        smry += sep
        smry += f"Elapsed time={self.seconds:.5f} seconds =====>>> {self.time_exe:%c}\n"
        smry += sep[:-1]
        return smry

    def summary(self):
        return self.as_txt()
        
    def as_dataframe(self):
        df = pd.DataFrame({
            "component": self.e_types,
            "parameter": self.est_names,
            "estimate": self.estimates,
            "stderr": self.stderrs,
            "wald": self.wald,
            "pvalue": self.pval
        })
        df.ffill(inplace=True)
        return df
    
    def save(self, fname):
        """
        Save a CUBresult object to file
        """
        filename = f"{fname}.cub.fit"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Fitting object saved to {filename}")

class CUBsample(object):
    def __init__(self, rv, m, pars,
        model, diss, theoric,
        par_names, sh=None,
        V=None, W=None, X=None,
        Y=None, Z=None,
        seed=None):
        self.model = model
        self.diss = diss
        self.theoric = theoric
        self.m = m
        self.sh = sh
        self.pars = np.array(pars)
        self.par_names = np.array(par_names)
        self.V = V
        self.W = W
        self.X = X
        self.Y = Y
        self.Z = Z
        self.p = self.pars.size
        self.rv = rv
        self.n  = rv.size
        self.seed = seed
        self.par_list = ""
        if self.sh is not None:
            self.par_list += f"sh={self.sh}; "
        for i in range(self.p):
            self.par_list += f"{self.par_names[i]}={self.pars[i]:.3f}"
            if i < (self.p-1):
                self.par_list += "; "

    def __str__(self):
        return f"CUBsample({self.model}; n={self.n}; {self.par_list})"

    def summary(self):
        smry = "=======================================================================\n"
        smry += f"=====>>> {self.model} model <<<===== Drawn random sample\n"
        smry += "=======================================================================\n"
        smry += f"m={self.m}  Sample size={self.n}  seed={self.seed}\n"
        par_rows = self.par_list.replace('; ','\n')
        smry += f"{par_rows}\n"
        smry += "=======================================================================\n"
        #smry += "Uncertainty\n"
        #smry += f"(1-pi) = {1-self.pi:.6f}\n"
        #smry += "Feeling\n"
        #smry += f"(1-xi) = {1-self.xi:.6f}\n"
        #smry += "=======================================================================\n"
        smry += "Sample metrics\n"
        smry += f"Mean     = {np.mean(self.rv):.6f}\n"
        smry += f"Variance = {np.var(self.rv, ddof=1):.6f}\n"
        smry += f"Std.Dev. = {np.std(self.rv, ddof=1):.6f}\n"
        smry += "-----------------------------------------------------------------------\n"
        smry += f"Dissimilarity = {self.diss:.7f}\n"
        smry += "======================================================================="
        return smry

    def plot(self, figsize=(7, 5),
        kind="bar", #options: scatter, bar
        ax=None, saveas=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        R = choices(self.m)
        f = freq(self.rv, self.m)
        if kind == "bar":
            ax.bar(R, f/self.rv.size, color="None",
                edgecolor="k", label="drawn")
        else:
            if kind != "scatter":
                print(f"WARNING: kind `{kind}` unknown. Using `scatter` instead.")
            ax.scatter(R, f/self.rv.size, facecolor="None",
                edgecolor="k", s=200, label="drawn")
        #p = pmf(self.m, self.pi, self.xi)
        ax.stem(R, self.theoric, linefmt="--r",
            markerfmt="none", label="generator")
        ax.set_xticks(R)
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability mass")
        #TODO: title too long for models with covariates
        ax.set_title(self)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        if saveas is not None:
            fig.savefig(saveas,
                bbox_inches='tight')
        else:
            return fig, ax

    def as_dataframe(self,
        varname="ordinal"):
        df = pd.DataFrame({
            varname: self.rv
        })
        return df

    def save(self, fname):
        """
        Save a CUBsample object to file
        """
        filename = f"{fname}.cub.sample"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Sample saved to {filename}")
