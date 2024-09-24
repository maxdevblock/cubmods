`cush` module Main Functions, Ancillary Functions and Classes

```Python
from cubmods import cush
```

See [cush_family](../05_cush_family.md) Manual for details about the models.

***

# Main Functions

## `.draw(m, sh, delta, n)`

Draws a random sample from a given CUBSH model without covariates.

- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `delta` (_float_): weight of shelter choice, must be $(0,1]$
  - `n` (_int_): number of observations to be drawn, must be $n>0$
  - `seed=None` (_int_): seed to ensure reproducibility

- Returns
  - an instance of `CUBsample` Class (#TODO: link)

## `.mle(sample, m, sh)`

Function to estimate and validate a CUB model without covariates for given ordinal responses. The function also checks if the estimated variance-covariance matrix is not positive definite and if `NaN` are produced during calculations: in case, it prints a warning message and returns a matrix and related results with `NaN` entries.

- Arguments
  - `sample` (_array_): the observed sample; can be a _list_ or a `numpy` _array_
  - `m` (_int_): number of ordinal responses; must be $m>3$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `ass_pars=None` (_dictionary_): if provided, a dictionary of a known model parameters
  - `maxiter=None` (_int_): for GEM compatibility
  - `tol=None` (_float_): for GEM compatibility
- Returns
  - an instance of `CUBresCUSH` Class [see here](cush.md#CUBresCUSH)

***

# Ancillary Functions

## `.pmf(m, sh, delta)`
probability distribution Function of a specified CUB model.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `delta` (_float_): weight of shelter choice, must be $(0,1]$
- Returns
  - an _array_ of $m$ elements, PMF of the specified model.

## `.loglik(sample, m, sh, delta)`
Loglikelihood of a specified CUB model given an observed sample.
- Arguments
  - `sample` (_array_): the observed sample of ordinal responses
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `delta` (_float_): weight of shelter choice, must be $(0,1]$
- Returns
  - the computed loglikelihood (_int_)

***

# Classes

## `CUBresCUSH`

Extension of the basic `CUBres` Class (#TODO: link). Is returned by `.mle()` function [see here](cub.md#mle).

- Methods
  - same of `CUBres` Class [see here]() #TODO: link

- Functions
  - `.plot_ordinal()`
    
    Plots the observed sample relative frequencies, the probability distribution of the estimated model and (if provided) the probability distribution of the kwown (generating) model.

    - Arguments
      - `kind="bar"` (_string_): how to plot the observed sample relative frequencies; options: `bar`, `scatter`
      - `ax=None` (_matplotlib ax_): subplot, if `None` a new figure will be created with specified `figsize`; see [matplotlib](https://matplotlib.org) documentation for details
      - `figsize=(7, 7)` (_tuple_): a tuple of integers of figure size `(weight, height)`; only effective if `ax=None`; see [matplotlib](https://matplotlib.org) documentation for details
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; only effective if `ax=None`; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - _ax_ if `ax` is not `None` otherwise a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - `.plot_estim()`

    Plots the asymptotic confidence ellipse of estimated parameters.
      
    - Arguments
      - `ax=None` (_matplotlib ax_): subplot, if `None` a new figure will be created with specified `figsize`; see [matplotlib](https://matplotlib.org) documentation for details
      - `figsize=(7, 7)` (_tuple_): a tuple of integers of figure size `(weight, height)`; only effective if `ax=None`; see [matplotlib](https://matplotlib.org) documentation for details
      - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
      - `magnified=False` (_boolean_): if `False` the axes limits will be the full parameter space; otherwise if `True` matplotlib will automatically adjust the limits to fit the ellipse
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; only effective if `ax=None`; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - _ax_ if `ax` is not `None` otherwise a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details
 
  - `.plot()`
    
    Default plot tool. Plots a figure with 3 rows and one column with `.plot_ordinal()`, `.plot_estim()` and `.plot_estim(magnified=True)`
    - Arguments
      - `figsize=(7, 15)` (_tuple_) a tuple of integers of figure size `(weight, height)`; see [matplotlib](https://matplotlib.org) documentation for details
      - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - other functions of `CUBres` Class [see here]() #TODO: link
