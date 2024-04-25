`cube_ywz` module Main Functions, Ancillary Functions and Classes

```Python
from cubmods import cube_ywz
```

See [cube_family](../04_cube_family.md) Manual for details about the models.

***

# Main Functions

## `.draw(m, n, beta, gamma, alpha, Y, W, Z)`

Draws a random sample from a given model.

- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `n` (_int_): number of observations to be drawn, must be $n>0$
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `alpha` (_array_): array of parameters $\pmb\alpha$ for Overdispersion covariates; must be of length `Z.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `Z` (_DataFrame_): a `numpy` DataFrame with covariates values for Overdispersion
  - `seed=None` (_int_): seed to ensure reproducibility

- Returns
  - an instance of `CUBsample` Class (#TODO: link)

## `.mle(m, sample, Y, W, Z,)`

Estimates parameters from an observed sample.

- Arguments
  - `sample` (_array_): the observed sample; can be a _list_ or a `numpy` _array_
  - `m` (_int_): number of ordinal responses
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty; column names will be taken as covariate variable names; it must not contain a column named `constant`
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling; column names will be taken as covariate variable names; it must not contain a column named `constant`
  - `Z` (_DataFrame_): a `numpy` DataFrame with covariates values for Overdispersion; column names will be taken as covariate variable names; it must not contain a column named `constant`
  - `gen_pars=None` (_dictionary_): if provided, a dictionary of a known model parameters `{"beta": <array>, "gamma": <array>, "alpha": <array>}`
  - `maxiter=1000` (_int_): maximum number of iterations for the EM algorithm
  - `tol=1e-2` (_float_): tolerance for the EM algorithm
- Returns
  - an instance of `CUBresCUBEYWZ` Class [see here](cub.md#CUBresCUBEYWZ)

***

# Ancillary Functions

## `.pmf(m, beta, gamma, alpha, Y, W, Z)`
Average Estimated Probability mass of a specified model.
- Arguments
  - `m` (_int_): number of ordinal responses
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `alpha` (_array_): array of parameters $\pmb\alpha$ for Overdispersion covariates; must be of length `Z.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `Z` (_DataFrame_): a `numpy` DataFrame with covariates values for Overdispersion
- Returns
  - an _array_ of $m$ elements, Average Estimated Probability of the specified model.

## `.pmfi()`
PMF of a specified model for each statistical unit $i$ given the covariates and the parameters.
- Arguments
  - `m` (_int_): number of ordinal responses
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `alpha` (_array_): array of parameters $\pmb\alpha$ for Overdispersion covariates; must be of length `Z.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `Z` (_DataFrame_): a `numpy` DataFrame with covariates values for Overdispersion
- Returns
  - an _matrix_ $n \times m$, PMF of the specified model for each statistical unit.

## `.loglik(m, sample, Y, W, Z, beta, gamma, alpha)`
Loglikelihood of a specified CUB model given an observed sample.
- Arguments
  - `sample` (_array_): the observed sample; can be a _list_ or a `numpy` _array_
  - `m` (_int_): number of ordinal responses
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `alpha` (_array_): array of parameters $\pmb\alpha$ for Overdispersion covariates; must be of length `Z.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `Z` (_DataFrame_): a `numpy` DataFrame with covariates values for Overdispersion
- Returns
  - the computed loglikelihood (_int_)

## `.varcov(m, sample, beta, gamma, alpha, Y, W, Z)`
Asymptotic covariance matrix of estimated parameters.
- Arguments
  - `sample` (_array_): the observed sample; can be a _list_ or a `numpy` _array_
  - `m` (_int_): number of ordinal responses
  - `beta` (_array_): array of parameters $\pmb\beta$ for Uncertainty covariates; must be of length `Y.columns+1`
  - `gamma` (_array_): array of parameters $\pmb\gamma$ for Feeling covariates; must be of length `W.columns+1`
  - `alpha` (_array_): array of parameters $\pmb\alpha$ for Overdispersion covariates; must be of length `Z.columns+1`
  - `Y` (_DataFrame_): a `numpy` DataFrame with covariates values for Uncertainty
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `Z` (_DataFrame_): a `numpy` DataFrame with covariates values for Overdispersion
- Returns
  - a matrix $u \times u$ where $u=|\pmb\beta|+|\pmb\gamma|+|\pmb\alpha|$ of the asymptotic covariance

## `.init_theta(m, sample, W, p, v)`
Initial values of $(\pmb\beta^{(0)}, \pmb\gamma^{(0)}, \pmb\alpha^{(0)})$ for EM algorithm.
- Arguments
  - `sample` (_array_): the observed sample
  - `m` (_int_): number of ordinal responses
  - `W` (_DataFrame_): a `numpy` DataFrame with covariates values for Feeling
  - `p` (_int_): number of covariates for Uncertainty
  - `v` (_int_): number of covariates for Overdispersion
- Returns
  - a _tuple_ of $(\pmb\beta^{(0)}, \pmb\gamma^{(0)}, \pmb\alpha^{(0)})$

***

# Classes

## `CUBresCUBEYWZ`

Extension of the basic `CUBres` Class (#TODO: link). Is returned by `.mle()` function [see here](cub.md#mle).

- Methods
  - same of `CUBres` Class [see here]() #TODO: link

- Functions
  - `.plot_ordinal()`
    
    Plots the observed sample relative frequencies, the average estimated probability mass of the estimated model.

    - Arguments
      - `kind="bar"` (_string_): how to plot the observed sample relative frequencies; options: `bar`, `scatter`
      - `ax=None` (_matplotlib ax_): subplot, if `None` a new figure will be created with specified `figsize`; see [matplotlib](https://matplotlib.org) documentation for details
      - `figsize=(7, 5)` (_tuple_): a tuple of integers of figure size `(weight, height)`; only effective if `ax=None`; see [matplotlib](https://matplotlib.org) documentation for details
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; only effective if `ax=None`; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - _ax_ if `ax` is not `None` otherwise a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - `.plot()`
    
    Default plot tool. Plots a figure with 1 rows and 1 column with `.plot_ordinal()`
    - Arguments
      - `figsize=(7, 5)` (_tuple_) a tuple of integers of figure size `(weight, height)`; see [matplotlib](https://matplotlib.org) documentation for details
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - other functions of `CUBres` Class [see here]() #TODO: link
  