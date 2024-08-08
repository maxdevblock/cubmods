`cubsh` module Main Functions, Ancillary Functions and Classes

```Python
from cubmods import cubsh
```

See [cub_family](../03_cubsh_family.md) Manual for details about the models.

***

# Main Functions

## `.draw(m, sh, pi1, pi2, xi, n)`

Draws a random sample from a given CUBSH model without covariates.

- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `pi1` (_float_): weight of shifted binomial component, must be $(0,1]$
  - `pi2` (_float_): weight of discrete uniform component, must be $(0,1]$
  - `xi` (_float_): parameter of Feeling $(1-\xi)$, must be $[0,1]$
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
  - `gen_pars=None` (_dictionary_): if provided, a dictionary of a known model parameters
  - `maxiter=500` (_int_): maximum number of iterations for the EM algorithm
  - `tol=1e-4` (_float_): tolerance for the EM algorithm
- Returns
  - an instance of `CUBresCUBSH` Class [see here](cubsh.md#CUBresCUBSH)

***

# Ancillary Functions

## `.pmf(m, sh, pi1, pi2, xi)`
probability distribution Function of a specified CUB model.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `pi1` (_float_): weight of shifted binomial component, must be $(0,1]$
  - `pi2` (_float_): weight of discrete uniform component, must be $(0,1]$
  - `xi` (_float_): parameter of Feeling $(1-\xi)$, must be $[0,1]$
- Returns
  - an _array_ of $m$ elements, PMF of the specified model.

## `.loglik(m, sh, pi1, pi2, xi, f)`
Loglikelihood of a specified CUB model given an observed sample.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `pi1` (_float_): weight of shifted binomial component, must be $(0,1]$
  - `pi2` (_float_): weight of discrete uniform component, must be $(0,1]$
  - `xi` (_float_): parameter of Feeling $(1-\xi)$, must be $[0,1]$
  - `f` (_array_): absolute frequencies of the observed model; must be of size $m$
- Returns
  - the computed loglikelihood (_int_)

## `.varcov(m, sh, pi1, pi2, xi, n)`
Asymptotic covariance matrix of estimated parameters.
- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
  - `pi1` (_float_): weight of shifted binomial component, must be $(0,1]$
  - `pi2` (_float_): weight of discrete uniform component, must be $(0,1]$
  - `xi` (_float_): parameter of Feeling $(1-\xi)$, must be $[0,1]$
  - `n` (_int_): the number of observations in the observed sample
- Returns
  - a matrix $3 \times 3$ of the estimated covariance

## `.init_theta(f, m, sh)`
Initial values of $(\pi^{(0)}, \xi^{(0)})$ for EM algorithm.
- Arguments
  - `f` (_array_): absolute frequencies of the observed model; must be of size $m$
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `sh` (_int_): shelter choice (must be $[1,m]$)
- Returns
  - a _tuple_ of $(\pi_1^{(0)}, \pi_2^{(0)}, \xi^{(0)})$

***

# Classes

## `CUBresCUBSH`

Extension of the basic `CUBres` Class (#TODO: link). Is returned by `.mle()` function [see here](cub.md#mle).

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

  - `.plot_confell()`

    Plots the asymptotic confidence ellipse of estimated parameters.
      
    - Arguments
      - `ax=None` (_matplotlib ax_): subplot, if `None` a new figure will be created with specified `figsize`; see [matplotlib](https://matplotlib.org) documentation for details
      - `figsize=(7, 5)` (_tuple_): a tuple of integers of figure size `(weight, height)`; only effective if `ax=None`; see [matplotlib](https://matplotlib.org) documentation for details
      - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
      - `magnified=False` (_boolean_): if `False` the axes limits will be the full parameter space; otherwise if `True` matplotlib will automatically adjust the limits to fit the ellipse
      - `equal=True` (_boolean_): if `True` the axes will be equally spaced `ax.set_aspect("equal")`; see [matplotlib](https://matplotlib.org) documentation for details
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; only effective if `ax=None`; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
      - `confell=False` (_boolean_): deprecated; whether to plot the $(\pi,\xi)$ confidence ellipse
      - `debug=False` (_boolean_): print debugging info for the confidence ellipse
    - Returns
      - _ax_ if `ax` is not `None` otherwise a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details
 
  - `.plot_3d(ax)`

    Plots the 3-dimensional confidence ellipsoid of estimated $(\pi, \xi, \delta)$.

    - Arguments
      - `ax=None` (_matplotlib ax_): subplot; see [matplotlib](https://matplotlib.org) documentation for details
      - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
      - `magnified=False` (_boolean_): if `False` the axes limits will be the full parameter space; otherwise if `True` matplotlib will automatically adjust the limits to fit the ellipsoid

  - `.plot()`
    
    Default plot tool. Plots a figure with 3 rows and one column with `.plot_ordinal()`, `.plot_3d()` and `.plot_3d(magnified=True)`
    - Arguments
      - `figsize=(7, 15)` (_tuple_) a tuple of integers of figure size `(weight, height)`; see [matplotlib](https://matplotlib.org) documentation for details
      - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
      - `confell=False` (_boolean_): deprecated
      - `debug=False` (_boolean_): deprecated
      - `test3=True` (_boolean_): deprecated
    - Returns
      - a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - other functions of `CUBres` Class [see here]() #TODO: link
