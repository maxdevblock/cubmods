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

See [cub module](./Reference%20Guide/cub.md) Reference Guide for more details.

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
- with `ass_pars` the parameter of a known model (if any) can be specified

```Python
# inferential method on drawn sample
mod = gem.from_formula(
    df=drawn.as_dataframe(),
    formula="ordinal~0|0|0",
    m=10,
    ass_pars={"pi": .7, "xi":.2}
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

### Example with covariates for Feeling component

See [cub_0w module](./Reference%20Guide/cub_0w.md) Reference Guide for more details.

```Python
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cubmods import cub_0w, gem

# load a dataset
df = pd.read_csv("https://maxpierini.it/cub/DIUBAS_DP.csv")

# MLE estimation with covariates for Feeling
mod = gem.from_formula(
    formula="disagiopersonale~0|"
    "frequenzadepressione+abbandonostudi|0",
    df=df
)
# print MLE summary
print(mod.summary())
# plot the results
mod.plot()
plt.show()
```

```
=======================================================================
=====>>> CUB(0W) model <<<===== ML-estimates
=======================================================================
m=10  Size=1243  Iterations=23  Maxiter=500  Tol=1E-04
-----------------------------------------------------------------------
Uncertainty
                      Estimates  StdErr     Wald  p-value
pi                       +0.891  0.0169   52.717   0.0000
-----------------------------------------------------------------------
Feeling
                      Estimates  StdErr     Wald  p-value
constant                  +2.33  0.0852   27.333   0.0000
frequenzadepressione      -0.37  0.0127  -29.194   0.0000
abbandonostudi           -0.058  0.0082   -7.074   0.0000
=======================================================================
Dissimilarity = 0.0915
Loglik(MOD)   = -2299.812
Loglik(uni)   = -2862.113
Mean-loglik   = -1.850
-----------------------------------------------------------------------
AIC = 4607.62
BIC = 4628.13
=======================================================================
Elapsed time=0.10926 seconds =====>>> Thu Apr 25 12:20:33 2024
=======================================================================
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/8a815c9a-b04e-4220-b6a3-f1e73212ba66)

```Python
# Draw a random sample
n = 1000
np.random.seed(1)
W1 = np.random.randint(1, 10, n)
np.random.seed(42)
W2 = np.random.randint(1, 10, n)
W = pd.DataFrame({
    "W1": W1, "W2": W2
})
drawn = cub_0w.draw(m=10, n=n, 
    pi=mod.estimates[0],
    gamma=mod.estimates[1:],
    W=W
)
drawn.plot()
plt.show()
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/89d308b8-85da-4810-b768-ada9d1204490)

```Python
# MLE estimation of random sample
W["ordinal"] = drawn.rv
mod1 = gem.from_formula(
    formula="ordinal~0|W1+W2|0",
    df=W
)
# Print MLE summary
print(mod1.summary())
# plot the results
mod1.plot()
plt.show()
```

```
=======================================================================
=====>>> CUB(0W) model <<<===== ML-estimates
=======================================================================
m=10  Size=1000  Iterations=23  Maxiter=500  Tol=1E-04
-----------------------------------------------------------------------
Uncertainty
          Estimates  StdErr     Wald  p-value
pi           +0.891  0.0186    47.99   0.0000
-----------------------------------------------------------------------
Feeling
          Estimates  StdErr     Wald  p-value
constant     +2.313  0.0872   26.529   0.0000
W1           -0.371  0.0118  -31.411   0.0000
W2           -0.054  0.0105   -5.149   0.0000
=======================================================================
Dissimilarity = 0.0576
Loglik(MOD)   = -1865.222
Loglik(uni)   = -2302.585
Mean-loglik   = -1.865
-----------------------------------------------------------------------
AIC = 3738.44
BIC = 3758.07
=======================================================================
Elapsed time=0.08926 seconds =====>>> Thu Apr 25 12:20:34 2024
=======================================================================
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/019fde25-8a19-41c2-87d8-f738e888b8fe)

### Example with covariates for Uncertainty component

See [cub_y0 module](./Reference%20Guide/cub_y0.md) Reference Guide for more details.

```Python
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cubmods import cub_y0, gem

# load a dataset
df = pd.read_csv("https://maxpierini.it/cub/DIUBAS_DP.csv")

# MLE estimation with covariates for Uncertainty
mod = gem.from_formula(
    formula="disagiopersonale~"
    "frequenzadepressione+abbandonostudi|0|0",
    df=df
)
# print MLE summary
print(mod.summary())
# plot the results
mod.plot()
plt.show()
```

```
=======================================================================
=====>>> CUB(Y0) model <<<===== ML-estimates
=======================================================================
m=10  Size=1243  Iterations=50  Maxiter=500  Tol=1E-04
-----------------------------------------------------------------------
Uncertainty
                      Estimates  StdErr    Wald  p-value
constant                -16.237  2.5935  -6.261   0.0000
frequenzadepressione     +2.166  0.3406    6.36   0.0000
abbandonostudi           +0.309  0.0849   3.644   0.0003
-----------------------------------------------------------------------
Feeling
                      Estimates  StdErr    Wald  p-value
xi                       +0.246  0.0078  31.398   0.0000
=======================================================================
Dissimilarity = 0.1445
Loglik(MOD)   = -2567.687
Loglik(uni)   = -2862.113
Mean-loglik   = -2.066
-----------------------------------------------------------------------
AIC = 5143.37
BIC = 5163.88
=======================================================================
Elapsed time=0.30304 seconds =====>>> Thu Apr 25 12:31:03 2024
=======================================================================
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/2d4dab80-ca06-4d98-9d97-6e92d7545f04)

```Python
# Draw a random sample
n = 1000
np.random.seed(1)
Y1 = np.random.randint(1, 10, n)
np.random.seed(42)
Y2 = np.random.randint(1, 10, n)
Y = pd.DataFrame({
    "Y1": Y1, "Y2": Y2
})
drawn = cub_y0.draw(m=10, n=n, 
    beta=mod.estimates[:-1], xi=mod.estimates[-1], Y=Y)
drawn.plot()
plt.show()
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/9a7bc632-5aa2-4a6e-b478-051966067b8f)

```Python
# MLE estimation of random sample
Y["ordinal"] = drawn.rv
mod1 = gem.from_formula(
    formula="ordinal~Y1+Y2|0|0",
    df=Y
)
# Print MLE summary
print(mod1.summary())
# plot the results
mod1.plot()
plt.show()
```

```
=======================================================================
=====>>> CUB(Y0) model <<<===== ML-estimates
=======================================================================
m=10  Size=1000  Iterations=60  Maxiter=500  Tol=1E-04
-----------------------------------------------------------------------
Uncertainty
          Estimates  StdErr    Wald  p-value
constant    -28.134  7.8829  -3.569   0.0004
Y1           +3.715  0.9908   3.749   0.0002
Y2           +0.594  0.2397   2.477   0.0132
-----------------------------------------------------------------------
Feeling
          Estimates  StdErr    Wald  p-value
xi           +0.247  0.0088  28.133   0.0000
=======================================================================
Dissimilarity = 0.0284
Loglik(MOD)   = -2145.126
Loglik(uni)   = -2302.585
Mean-loglik   = -2.145
-----------------------------------------------------------------------
AIC = 4298.25
BIC = 4317.88
=======================================================================
Elapsed time=0.30413 seconds =====>>> Thu Apr 25 12:31:03 2024
=======================================================================
```

![image](https://github.com/maxdevblock/cubmods/assets/46634650/f3a0928a-08f1-4ff2-9e4c-c1f0568bce8c)

