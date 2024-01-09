import numpy as np
from scipy.optimize import minimize
from .general import (
    choices
)

def pmf(pars, m, shs):
    R = choices(m)
    p = np.zeros(m)
    for i, sh in enumerate(shs):
        p += (R==sh)*pars[i]
    p += (1-np.sum(pars))/m
    return p

def draw(pars, m, shs, n, seed=None):
    R = choices(m)
    p = pmf(pars, m, shs)
    np.random.seed(seed)
    rv = np.random.choice(
        R, size=n,
        p=p
    )
    return rv

def prob(pars, sample, m, shs):
    nc = shs.size
    n = sample.size
    fc = np.zeros(nc)
    D = np.ndarray(shape=(nc,n))
    for i in range(nc):
        fc[i] = (sample==shs[i]).sum()/n
    fcz = 1-np.sum(fc)
    paz = 1-np.sum(pars)
    p = n*(np.sum(
        [pars[i]*fc[i] for i in range(nc)]
        )+fcz*paz)
    return p

def loglik(pars, sample, m, shs):
    p = prob(pars, sample, m, shs)
    l = np.sum(np.log(p))
    return l

def effe(pars, sample, m, shs):
    nc = shs.size
    n = sample.size
    fc = np.zeros(nc)
    #D = np.ndarray(shape=(nc,n))
    for i in range(nc):
        fc[i] = (sample==shs[i]).sum()/n
    fcz = 1-np.sum(fc)
    paz = 1-np.sum(pars)
    l = n*(
        np.sum([
            fc[i]*np.log(pars[i]-paz/m)
        ]) + fcz*np.log(paz/m)
    )
    return -l

def mle(sample, m, shs,
    maxiter=None, tol=None):
    shs = np.array(shs)
    n = sample.size
    nc = shs.size
    if nc >= (m-1):
        return None
    fc = np.zeros(nc)
    pars0 = np.zeros(nc)
    for i in range(nc):
        fc[i] = (sample==shs[i]).sum()/n
    for i in range(nc):
        fcu = np.sum(fc)-fc[i]
        pars0[i] = max([
            .01, ((m-(nc-1))*fc[i]+fcu-1)/(m-nc)
        ])
    bounds = [(.01,.99) for i in range(nc)]
    est = minimize(
        effe, x0=pars0,
        bounds=bounds,
        args=(sample, m, shs)
    )
    deltas = est.x
    return deltas
    