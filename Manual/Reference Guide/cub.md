`cub` module Main Functions, Ancillary Functions and Classes

***

# Main Functions

## `.draw()`

Draws a random sample from a given model.

- Arguments
  - `m` (_int_): number of ordinal responses; the support of random variable will be $[1,m]$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
  - `n` (_int_): number of observations to be drawn, must be $n>0$
  - `seed=None` (_int_): seed to ensure reproducibility

- Returns
  - an instance of `CUBsample` Class (#TODO: link)

## `.mle()`

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

## `.pmf()`
Probability Mass Function of a specified CUB model.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
- Returns
  - an _array_ of $m$ elements, PMF of the specified model.

## `.loglik()`
Loglikelihood of a specified CUB model given an observed sample.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
  - `f` (_array_): absolute frequencies of the observed model; must be of size $m$
- Returns
  - the computed loglikelihood (_int_)

## `.varcov()`
Estimated ovariance matrix of estimated parameters.
- Arguments
  - `m` (_int_): number of ordinal responses; should be $m>3$
  - `pi` (_float_): parameter of Uncertainty $(1-\pi)$, must be $(0,1]$
  - `xi` (_float_): parameters of Feeling $(1-\xi)$, must be $[0,1]$
  - `ordinal` (_array_): the observed sample
- Returns

## `.init_theta()`
Initial values of $(\pi^{(0)}, \xi^{(0)})$ for EM algorithm.
- Arguments
  - `f` (_array_): absolute frequencies of the observed model; must be of size $m$
  - `m` (_int_): number of ordinal responses; should be $m>3$
- Returns
  - a _tuple_ of $(\pi^{(0)}, \xi^{(0)})$

***

# Classes

## `CUBresCUB00`

Extension of the basic `CUBres` Class (#TODO: link). Is return by `.mle()` function [see here](cub.md#mle).

- Methods
  - `.name`: returns what _type_, description

- Functions
  - `.name()`
    - Arguments
      - `name` (_type_): description, options
      - `name=default` (_tyoe_): description, option
    - Returns
      - _type_: description (linkto)
  - `.name()`