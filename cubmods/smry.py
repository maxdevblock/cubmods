#import datetime as dt
import pickle
import numpy as np
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
    par_names = np.asarray(est_names)
    par_types = np.asarray(e_types)
    lparnames = len(max(par_names, key=len))
    pars = np.asarray(estimates)
    pars = np.round(pars, 3)
    #print(pars)
    lpars = len(max(pars.astype(str), key=len))
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
    lpvals = len(max(pvals.astype(str), key=len))
    space4 = 2
    
    sep = "=======================================================================\n"
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
            smry += sep
            smry += f"{par_types[i]}\n"
            smry += est
        spaceA = (space1+9)-len(par_names[i])-(len(f"{pars[i]:+}"))
        spaceB = space2+6-len(str(ses[i]))
        spaceC = space3+4-len(str(walds[i]))
        spaceD = space4+7-6
        #print(f"`{str(pars[i])}`")
        smry += f"{par_names[i]}{' '*spaceA}{pars[i]:+}{' '*spaceB}{ses[i]}{' '*spaceC}{walds[i]}{' '*spaceD}{pvals[i]:.4f}"
        smry += "\n"
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
        l_ = "* Saturated model without covariates\n"
    if logliksatcov is not None:
        c_ = "^ not valid for continuous covariates\n"
        smry += f"Logl(satcov)^ = {logliksatcov:.3f}\n"
    
    warn = " (!)" if logliksat<loglike else ""
    smry += f"Loglike(sat){ls}= {logliksat:.3f}{warn}\n"
    smry += f"Loglike(MOD)  = {loglike:.3f}\n"
    smry += f"Loglike(uni)  = {loglikuni:.3f}\n"
    smry += f"Mean-loglike  = {muloglik:.3f}\n"
    smry += f"Deviance{ls}    = {dev:.3f}{warn}\n"
    if c_ is not None:
        smry += c_
    if l_ is not None:
        smry += l_
    if rho is not None:
        smry += f"Correlation   = {rho:.4f}\n"
    smry += sep
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
        loglikuni, logliksat,
        dev, AIC, BIC,
        seconds, time_exe,
        # optional parameters
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

    def summary(self):
        return as_txt(
            **self.__dict__
        )
    
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

    def plot(self, figsize=(7, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        R = choices(self.m)
        f = freq(self.rv, self.m)
        ax.scatter(R, f/self.rv.size, facecolor="None",
            edgecolor="k", s=200, label="drawn")
        #p = pmf(self.m, self.pi, self.xi)
        ax.stem(R, self.theoric, linefmt="--r",
            markerfmt="none", label="generator")
        ax.set_xticks(R)
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_xlabel("Options")
        ax.set_ylabel("Probability mass")
        ax.set_title(self)
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))
        return fig

    def save(self, fname):
        """
        Save a CUBsample object to file
        """
        filename = f"{fname}.cub.sample"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Sample saved to {filename}")
