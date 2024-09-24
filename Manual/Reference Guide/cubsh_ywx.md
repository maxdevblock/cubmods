`cubsh_ywx` module Main Functions, Ancillary Functions and Classes

```Python
from cubmods import cubsh_ywx
```

See [cub_family](../03_cubsh_family.md) Manual for details about the models.

***

# Main Functions

## `.draw(m, n, sh, beta, gamma, omega, Y, W, X)`

Draws a random sample from a given CUBSH model without covariates.

- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `n` (_int_): number of observations to be drawn, must be $n>0$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `omega` (_array_): array of parameters $\pmb\gamma$ for Shelter covariates; must be of length `X.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `X` (_DataFrame_): a `numpy` DataFrame with covariates values for Shelter
  - `seed=None` (_int_): seed to ensure reproducibility

- Returns
  - an instance of `CUBsample` Class (#TODO: link)

## `.mle(m, sample, sh, Y, W, X)`

Function to estimate and validate a CUB model without covariates for given ordinal responses. The function also checks if the estimated variance-covariance matrix is not positive definite and if `NaN` are produced during calculations: in case, it prints a warning message and returns a matrix and related results with `NaN` entries.

- Arguments
  - `sample` (_array_): the observed sample; can be a _list_ or a `numpy` _array_
  - `m` (_int_): number of ordinal responses; must be $m>3$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `X` (_DataFrame_): a `numpy` DataFrame with covariates values for Shelter
  - `ass_pars=None` (_dictionary_): if provided, a dictionary of a known model parameters
  - `maxiter=500` (_int_): maximum number of iterations for the EM algorithm
  - `tol=1e-4` (_float_): tolerance for the EM algorithm
- Returns
  - an instance of `CUBresCUBSHYWX` Class [see here](cubsh_ywx.md#CUBresCUBSHYWX)

***

# Ancillary Functions

## `.pmf(m, sh, beta, gamma, omega, Y, W, X)`
probability distribution Function of a specified CUB model.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `omega` (_array_): array of parameters $\pmb\gamma$ for Shelter covariates; must be of length `X.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `X` (_DataFrame_): a `numpy` DataFrame with covariates values for Shelter
- Returns
  - an _array_ of $m$ elements, PMF of the specified model.

## `.loglik(m, sample, sh, Y, W, X, beta, gamma, omega)`
Loglikelihood of a specified CUB model given an observed sample.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sample` (_array_): the observed sample of ordinal responses
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `omega` (_array_): array of parameters $\pmb\gamma$ for Shelter covariates; must be of length `X.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `X` (_DataFrame_): a `numpy` DataFrame with covariates values for Shelter
- Returns
  - the computed loglikelihood (_int_)

## `.varcov(sample, m, sh, Y, W, X, beta, gamma, omega)`
Asymptotic covariance matrix of estimated parameters.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sample` (_array_): the observed sample of ordinal responses
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `omega` (_array_): array of parameters $\pmb\gamma$ for Shelter covariates; must be of length `X.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `X` (_DataFrame_): a `numpy` DataFrame with covariates values for Shelter
- Returns
  - a matrix of the estimated covariance

## `.init_theta(m, sample, p, s, W)`
Initial values of $(\pi^{(0)}, \xi^{(0)})$ for EM algorithm.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `sample` (_array_): the observed sample of ordinal responses
  - `p` (_int_): number of `Y` covariates for Uncertainty
  - `s` (_int_): number of `X` covariates for Shelter Choice
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
- Returns
  - a _tuple_ of $(\pmb\beta^{(0)}, \pmb\gamma^{(0)}, \pmb\omega^{(0)})$

***

# Classes

## `CUBresCUBSHYWX`

Extension of the basic `CUBres` Class (#TODO: link). Is returned by `.mle()` function [see here](cubsh_ywx.md#mle).

- Methods
  - same of `CUBres` Class [see here]() #TODO: link

- Functions
  - `.plot_ordinal()`
    
    Plots the observed sample relative frequencies, the probability distribution of the estimated model and (if provided) the probability distribution of the kwown (generating) model.

    - Arguments
      - `kind="bar"` (_string_): how to plot the observed sample relative frequencies; options: `bar`, `scatter`
      - `ax=None` (_matplotlib ax_): subplot, if `None` a new figure will be created with specified `figsize`; see [matplotlib](https://matplotlib.org) documentation for details
      - `figsize=(7, 5)` (_tuple_): a tuple of integers of figure size `(weight, height)`; only effective if `ax=None`; see [matplotlib](https://matplotlib.org) documentation for details
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; only effective if `ax=None`; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - _ax_ if `ax` is not `None` otherwise a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - `.plot()`
    
    Default plot tool. Plots a figure with `.plot_ordinal()`
    - Arguments
      - `figsize=(7, 5)` (_tuple_) a tuple of integers of figure size `(weight, height)`; see [matplotlib](https://matplotlib.org) documentation for details
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - other functions of `CUBres` Class [see here]() #TODO: link
