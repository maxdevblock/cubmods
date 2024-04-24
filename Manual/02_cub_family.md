# CUB family

***

## Without covariates

A model of the CUB family for responses with $m$ choices, without covariates is specified as

$$
\Pr(R=r|\boldsymbol{\theta}) = \pi \dbinom{m-1}{r-1}(1-\xi)^{r-1}\xi^{m-r}+\dfrac{1-\pi}{m}
$$

where $\pi$ and $\xi$ are the parameters for respectively the _uncertainty_ and the _feeling_ components.

Note that $(1-\pi)$ is the _Uncertainty_ weight and $(1-\xi)$ is the _Feeling_ component for
usual _positive wording_.

### Draw a sample

In the following example, a sample from a CUB model of $n=500$ observations of an ordinal variable with $m=10$ choices
and parameters $(\pi=.7, \xi=.2)$ 
will be drawn. A `seed=1` will be set to ensure reproducibility.

```Python
# import libraries
import matplotlib.pyplot as plt
from cubmods import cub, gem

# draw a sample
drawn = cub.draw(m=10, pi=.7, xi=.2,
                 n=500, seed=1)
# print the summary of the drawn sample
print(drawn.summary())
# show the plot of the drawn sample
drawn.plot()
plt.show()
```

The following results will be obtained.

```
=======================================================================
=====>>> CUB model <<<===== Drawn random sample
=======================================================================
m=10  Sample size=500  seed=1
pi=0.700
xi=0.200
=======================================================================
Sample metrics
Mean     = 7.368000
Variance = 5.687952
Std.Dev. = 2.384943
-----------------------------------------------------------------------
Dissimilarity = 0.0650938
=======================================================================
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/e2feeda5-8f06-4757-a430-9708d50ff317)

The `Dissimilarity` indicates the percentage of the sample that should be changed to
afford a perfect fit to the given model.

### Estimate parameters

Using the previously drawn sample, in the next example the parameters $(\hat\pi, \hat\xi)$ will be estimated.

Note that in the function `gem.from_formula`:
- `df` needs to be a `pandas` DataFrame; the function `drawn.as_dataframe()` will return a DataFrame with `ordinal` as default column name
- `formula` specifies the ordinal variable (`ordinal` in this case) and the covariates for each component (none in this case, so "0|0|0")
- if `m` is not provided, the maximum observed value will be taken
- with `gen_pars` the parameter of a known model (if any) can be specified

```Python
# inferential method on drawn sample
mod = gem.from_formula(
    df=drawn.as_dataframe(),
    formula="ordinal~0|0|0",
    m=10,
    gen_pars={"pi": .7, "xi":.2}
)
# print the summary of MLE
print(mod.summary())
# show the plot of MLE
mod.plot()
plt.show()
```

The script will produce the following output.

```
=======================================================================
=====>>> CUB00 model <<<===== ML-estimates
=======================================================================
m=10  Size=500  Iterations=13  Maxiter=500  Tol=1E-04
-----------------------------------------------------------------------
Uncertainty
    Estimates  StdErr    Wald  p-value
pi     +0.675   0.034  19.872   0.0000
-----------------------------------------------------------------------
Feeling
    Estimates  StdErr    Wald  p-value
xi     +0.188   0.009  20.808   0.0000
-----------------------------------------------------------------------
Correlation   = 0.2105
=======================================================================
Dissimilarity = 0.0599
Loglik(sat)   = -994.063
Loglik(MOD)   = -1000.111
Loglik(uni)   = -1151.293
Mean-loglik   = -2.000
Deviance      = 12.096
-----------------------------------------------------------------------
AIC = 2004.22
BIC = 2012.65
=======================================================================
Elapsed time=0.00187 seconds =====>>> Wed Apr 24 11:27:35 2024
=======================================================================
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/ca613509-a463-49ad-8f50-3f0bfd19c7ab)

See [cub module](./Reference%20Guide/cub.md) Reference Guide for more details.

***

## With covariates

A model of the CUB family for responses with $m$ choices, with covariates is specified as

$$
\Pr(R_i=r|\pmb{\theta}, \pmb{x}_i, \pmb{w}_i) = \pi_i \dbinom{m-1}{r-1}(1-\xi_i)^{r-1}\xi_i^{m-r}+\dfrac{1-\pi_i}{m}  
$$

$$
\begin{array}{l}
        \mathrm{logit}(\pi_i)=\pmb \beta \pmb{x}_i
        \\
        \mathrm{logit}(\xi_i)=\pmb \gamma \pmb{w}_i
\end{array}
$$
