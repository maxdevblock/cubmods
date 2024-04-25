`cub` module Main Functions, Ancillary Functions and Classes

```Python
from cubmods import cub
```

See [cub_family](../02_cub_family.md) Manual for details about the models.

***

# Main Functions

## `.draw(m, pi, xi, n)`

Draws a random sample from a given model.

- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
  - `n` (_int_): number of observations to be drawn, must be $n>0$
  - `seed=None` (_int_): seed to ensure reproducibility

- Returns
  - an instance of `CUBsample` Class (#TODO: link)

## `.mle(sample, m)`

Estimates parameters from an observed sample.

- Arguments
  - `sample` (_array_): the observed sample; can be a _list_ or a `numpy` _array_
  - `m` (_int_): number of ordinal responses; must be $m>3$
  - `gen_pars=None` (_dictionary_): if provided, a dictionary of a known model parameters `{"xi": <float>, "pi": <float>}`
  - `maxiter=500` (_int_): maximum number of iterations for the EM algorithm
  - `tol=1e-4` (_float_): tolerance for the EM algorithm
- Returns
  - an instance of `CUBresCUB00` Class [see here](cub.md#CUBresCUB00)

***

# Ancillary Functions

## `.pmf(m, pi, xi)`
Probability Mass Function of a specified CUB model.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
- Returns
  - an _array_ of $m$ elements, PMF of the specified model.

## `.loglik(m, pi, xi, f)`
Loglikelihood of a specified CUB model given an observed sample.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
  - `f` (_array_): absolute frequencies of the observed model; must be of size $m$
- Returns
  - the computed loglikelihood (_int_)

## `.varcov(m, pi, xi, ordinal)`
Asymptotic covariance matrix of estimated parameters.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
  - `ordinal` (_array_): the observed sample
- Returns
  - a matrix $2 \times 2$ of the estimated covariance

## `.init_theta(f, m)`
Initial values of $(\pi^{(0)}, \xi^{(0)})$ for EM algorithm.
- Arguments
  - `f` (_array_): absolute frequencies of the observed model; must be of size $m$
  - `m` (_int_): number of ordinal responses; should be $m>3$
- Returns
  - a _tuple_ of $(\pi^{(0)}, \xi^{(0)})$

***

# Classes

## `CUBresCUB00`

Extension of the basic `CUBres` Class (#TODO: link). Is returned by `.mle()` function [see here](cub.md#mle).

- Methods
  - same of `CUBres` Class [see here]() #TODO: link

- Functions
  - `.plot_ordinal()`
    
    Plots the observed sample relative frequencies, the probability mass of the estimated model and (if provided) the probability mass of the kwown (generating) model.

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
    - Returns
      - _ax_ if `ax` is not `None` otherwise a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - `.plot()`
    
    Default plot tool. Plots a figure with 3 rows and one column with `.plot_ordinal()`, `.plot_confell()` and `.plot_confell(magnified=True)`
    - Arguments
      - `figsize=(7, 15)` (_tuple_) a tuple of integers of figure size `(weight, height)`; see [matplotlib](https://matplotlib.org) documentation for details
      - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
      - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
      - a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details

  - other functions of `CUBres` Class [see here]() #TODO: link
