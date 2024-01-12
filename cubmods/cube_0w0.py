#TODO: implement
import datetime as dt
import numpy as np
from scipy.optimize import minimize
import scipy.stats as sps
from statsmodels.tools.numdiff import approx_hess
import matplotlib.pyplot as plt
from .general import (
    logis, dissimilarity,
    aic, bic, luni, lsat,
    freq, choices, lsatcov,
    #addones, colsof,
)
from .cube import (
    betar,
    #init_theta as ini_cube,
    mle as mle_cube
)
from .cub_0w import init_gamma
from .smry import CUBres, CUBsample

def pmfi(m, pi, gamma, phi, W):
    n = W.shape[0]
    xi = logis(W, gamma)
    p = np.ndarray(shape=(n, m))
    for i in range(n):
        pBe = betar(m=m, xi=xi[i], phi=phi)
        p[i,:] = pi*(pBe-1/m) + 1/m
    return p

def pmf(m, pi, gamma, phi, W):
    p = pmfi(m, pi, gamma, phi, W).mean(
        axis=0)
    #print(p_i)
    return p

def betabinomialxi(m, sample, xivett, phi):
    n = sample.size
    betabin = np.repeat(np.nan, n)
    for i in range(n):
        bebeta = betar(m=m, xi=xivett[i],
        phi=phi)
        betabin[i] = bebeta[sample[i]-1]
    return np.array(betabin)

def draw(m, n, pi, gamma, phi, W, seed=None):
    """
    generate random sample from CUB model
    """
    #np.random.seed(seed)
    assert n == W.shape[0]
    rv = np.repeat(np.nan, n)
    theoric_i = pmfi(m=m, pi=pi,
        gamma=gamma, phi=phi, W=W)
    #print("n", n)
    for i in range(n):
        if seed is not None:
            np.random.seed(seed*i)
        rv[i] = np.random.choice(
            choices(m=m),
            size=1,
            replace=True,
            p=theoric_i[i]
        )
    f = freq(m=m, sample=rv)
    theoric = pmf(m=m, pi=pi,
        gamma=gamma, phi=phi, W=W)
    diss = dissimilarity(f/n, theoric)
    pars = np.concatenate((
        [pi], gamma, [phi]
    ))
    par_names = np.concatenate((
        ["pi"],
        ["constant"],
        W.columns,
        ["phi"]
    ))
    sample = CUBsample(
        model="CUBE(0W0)",
        rv=rv.astype(int), m=m,
        pars=pars, par_names=par_names,
        seed=seed, W=W, diss=diss,
        theoric=theoric
    )
    return sample

def prob(m, sample, W, pi, gamma, phi):
    xivett = logis(Y=W, param=gamma)
    p = pi*(betabinomialxi(m=m, sample=sample,
        xivett=xivett, phi=phi)-1/m)+1/m
    return p

def loglik(m, sample, W, pi, gamma, phi):
    p = prob(m=m, sample=sample, W=W,
        pi=pi, gamma=gamma, phi=phi)
    l = np.sum(np.log(p))
    return l

def init_theta(m, sample, W, maxiter, tol):
    gamma = init_gamma(m=m, sample=sample,
        W=W)
    res_cube = mle_cube(m=m, sample=sample,
        maxiter=maxiter, tol=tol)
    pi = res_cube.estimates[0]
    phi = res_cube.estimates[2]
    return pi, gamma, phi

def effe(pars, sample, W, m):
    pi = pars[0]
    gamma = pars[1:-1]
    phi = pars[-1]
    l = loglik(m, sample, W, pi, gamma, phi)
    return -l

def mle(sample, m, W,
    gen_pars=None,
    maxiter=1000, tol=1e-6):
    start = dt.datetime.now()
    n = sample.size
    pi, gamma, phi = init_theta(
        m=m, sample=sample, W=W,
        maxiter=maxiter, tol=tol
    )
    l = loglik(m, sample, W, pi, gamma, phi)
    pars0 = np.concatenate((
        [pi], gamma, [phi]
    ))
    #print(pars0)
    q = gamma.size - 1
    bounds = [(.01, .99)]
    for _ in range(q+1):
        bounds.append((None, None))
    bounds.append((.01, .3))
    optim = minimize(effe, x0=pars0,
        args=(sample, W, m),
        method="L-BFGS-B",
        bounds=bounds
        )
    pars = optim.x
    pi = pars[0]
    gamma = pars[1:-1]
    phi = pars[-1]
    
    infmat = approx_hess(pars, effe,
        args=(sample, W, m))
    varmat = np.ndarray(shape=(pars.size,pars.size))
    varmat[:] = np.nan
    if np.any(np.isnan(infmat)):
        print("WARNING: NAs produced in information matrix")
    elif np.linalg.det(infmat) <= 0:
        print("ATTENTION: information matrix NOT positive definite")
    else:
        # varmat = np.linalg.inv(infmat)/n
        varmat = np.linalg.inv(infmat)
    
    stderrs = np.sqrt(np.diag(varmat))
    l = loglik(m, sample, W, pi, gamma, phi)
    theoric = pmf(m=m, pi=pi, gamma=gamma,
        phi=phi, W=W)
    f = freq(sample=sample, m=m)
    diss = dissimilarity(f/n, theoric)
    loglikuni = luni(m=m, n=n)
    logliksat = lsat(n=n, f=f)
    logliksatcov = lsatcov(
        sample=sample,
        covars=[W]
    )
    muloglik = l/n
    dev = 2*(logliksat-l)
    
    estimates = pars
    wald = estimates/stderrs
    pval = 2*(sps.norm().sf(abs(wald)))
    est_names = np.concatenate((
        ["pi"],
        np.concatenate((
            ["constant"],
            [x for x in W.columns]
        )),
        ["phi"]
    ))
    e_types = np.concatenate((
        ["Uncertainty"],
        ["Feeling"],
        np.repeat(None, q),
        ["Overdisperson"]
    ))
    AIC = aic(p=estimates.size, l=l)
    BIC = bic(l=l, p=estimates.size, n=n)
    end = dt.datetime.now()
    
    return CUBresCUBE0W0(
        model="CUBE(0W0)",
        n=n, m=m, sample=sample, f=f,
        estimates=estimates,
        stderrs=stderrs,
        wald=wald, pval=pval,
        est_names=est_names,
        e_types=e_types,
        theoric=theoric,
        AIC=AIC, BIC=BIC,
        loglike=l,
        logliksat=logliksat,
        logliksatcov=logliksatcov,
        loglikuni=loglikuni,
        muloglik=muloglik,
        diss=diss, dev=dev,
        varmat=varmat,
        seconds=(end-start).total_seconds(),
        time_exe=start,
        W=W
    )

class CUBresCUBE0W0(CUBres):
    
    def plot_ordinal(self,
        figsize=(7, 5),
        ax=None,
        saveas=None
        ):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize
            )
        #pi = self.estimates[0]
        #xi = self.estimates[1]
        #phi = self.estimates[2]
        title = f"{self.model} model    "
        title += f"$n={self.n}$\n"
        #title += fr"Estim($\pi={pi:.3f}$ , $\xi={xi:.3f}$ , $\phi={phi:.3f}$)"
        title += f"    Dissim(est,obs)={self.diss:.3f}"
        #TODO: add dissimilarity from generating model
        # if self.diss_gen is not None:
        #     title += "\n"
        #     title += fr"Gener($\pi={self.pi_gen:.3f}$ , $\xi={self.xi_gen:.3f}$)"
        #     title += f"    Dissim(est,gen)={self.diss_gen:.6f}"
        ax.set_title(title)

        R = choices(self.m)
        ax.set_xticks(R)
        ax.set_xlabel("Ordinal")
        ax.set_ylabel("Probability mass")

        ax.plot(R, self.theoric, ".b:",
            label="estimated", ms=10)
        ax.scatter(R, self.f/self.n, 
            facecolor="None",
            edgecolor="k", s=200,
            label="observed")
        if self.gen_pars is not None:
            pi_gen = self.gen_pars["pi"]
            gamma_gen = self.gen_pars["gamma"]
            phi_gen = self.gen_pars["phi"]
            p_gen = pmf(m=self.m, pi=pi_gen,
                gamma=gamma_gen, phi=phi_gen,
                W=self.W)
            ax.stem(R, p_gen, linefmt="--r",
            markerfmt="none", label="generating")

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.legend(loc="upper left",
            bbox_to_anchor=(1,1))

        if ax is None:
            if saveas is not None:
                fig.savefig(saveas, bbox_inches='tight')
            else:
                return fig, ax
        else:
            return ax
    
    def plot(self,
        ci=.95,
        saveas=None,
        figsize=(7, 5)
        ):
        """
        plot CUB model fitted from a sample
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_ordinal(ax=ax)
        #self.plot_confell(ci=ci, ax=ax[1])
        #self.plot_confell(
        #    ci=ci, ax=ax[2],
        #    magnified=True, equal=False)
        #plt.subplots_adjust(hspace=.25)
        if saveas is not None:
            fig.savefig(saveas, bbox_inches='tight')
        return fig, ax
    