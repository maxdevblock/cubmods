# cubmods
Statistical package: CUB models for ordinal responses.

This package is a _first Python implementation_ of statistical methods for the models of the class CUB, proposed by Professor Domenico Piccolo, 2003.

It contains inferential methods for each family of the class CUB (with or without covariates), basic graphical tools, and methods to draw random samples from given models.

It has been implemented by Massimo Pierini in 2024. It is mainly based upon the `CUB` package in `R`, implemented and maintened by Prof. Domenico Piccolo and Prof.ssa Rosaria Simone.

***

## Requirements

```
pip install -U numpy pandas scipy statsmodels
```

## Installation

Currently, the package must be downloaded from GitHub, unzipped, and placed in the same directory of the scripts that import it. 

Alternatively, you can install/update directly from GitHub. First, you need to have `git` installed (see [Install Git](https://github.com/git-guides/install-git) for detailed instructions for Windows, macOS and Linux). After installing `git` you'll need to restart your computer (to update the PATHs) and then you can install `cubmods` from GitHub with

```
pip install -U git+https://github.com/maxdevblock/cubmods.git@main
```

All these `pip` commands, can be run in `Spyder 6.x` IPython console.

When the next version will be complete, it will be uploaded on `PyPI`.

## Basic usage
```Python
# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(formula="ordinal ~ 0 | 0",
             m=10, pi=.7, xi=.2,
             n=500, seed=1)
print(drawn.summary())
drawn.plot()
plt.show()

# inferential method on drawn sample
mod = estimate(
    df=drawn.df,
    formula="ordinal~0|0",
    m=10,
    ass_pars={"pi": .7, "xi":.2}
)
print(mod.summary())
mod.plot()
plt.show()
```

## Manual
The following is a preliminary Manual and Reference Sheet.

- [Manual](https://cubmods.readthedocs.io/en/latest/) for models' specifications and working examples

## References
  - D'Elia A. (2003). Modelling ranks using the inverse hypergeometric distribution, Statistical Modelling: an International Journal, 3, 65--78
  - D'Elia A. and Piccolo D. (2005). A mixture model for preferences data analysis, Computational Statistics & Data Analysis},  \bold{49, 917--937
  - Capecchi S. and Piccolo D. (2017). Dealing with heterogeneity in ordinal responses, Quality and Quantity, 51(5), 2375--2393
  - Iannario M. (2014). Modelling Uncertainty and Overdispersion in Ordinal Data, Communications in Statistics - Theory and Methods, 43, 771--786
  - Piccolo D. (2015). Inferential issues for CUBE models with covariates, Communications in Statistics. Theory and Methods, 44(23), 771--786.
  - Iannario M. (2015). Detecting latent components in ordinal data with overdispersion by means of a mixture distribution, Quality & Quantity, 49, 977--987
  - Iannario M. and Piccolo D. (2016a). A comprehensive framework for regression models of ordinal data. Metron, 74(2), 233--252.
  - Iannario M. and Piccolo D. (2016b). A generalized framework for modelling ordinal data. Statistical Methods and Applications, 25, 163--189.

## Credits
@Author:      Massimo Pierini

@Date:        2023-24

@ThanksTo:    Domenico Piccolo, Rosaria Simone

@Contacts:    cub@maxpierini.it
