# cubmods
Statistical package: CUB models for ordinal responses.

This package is the first implementation of statistical methods for the models of the class CUB (proposed by Professor Domenico Piccolo, 2003) in Python.

It contains inferential methods for each family of the class CUB (with or without covariates), basic graphical tools, and methods to draw random samples from given models.

It has been implemented by Massimo Pierini, graduand in _Statistics & Big Data_ (L41) at _Universitas Mercatorum_ 2024. It is mainly based upon the `CUB` package in `R`, implemented and maintened by Prof. Domenico Piccolo and Prof.ssa Rosaria Simone.

***

## Requirements

```
pip install -U numpy pandas scipy statsmodels
```

Currently, the package must be downloaded from GitHub, unzipped, and placed in the same directory of the scripts that import it or you can install/update directly from GitHub using

```
pip install -U git+https://github.com/maxdevblock/cubmods.git@main
```

When the version `0.0.1` will be complete, it will be uploaded on `PyPI`.

## Basic usage
```Python
# import libraries
import matplotlib.pyplot as plt
from cubmods import cub, gem

# draw a sample
drawn = cub.draw(m=10, pi=.7, xi=.2,
                 n=500, seed=1)
print(drawn.summary())
drawn.plot()
plt.show()

# inferential method on drawn sample
mod = gem.from_formula(
    df=drawn.as_dataframe(),
    formula="ordinal~0|0|0",
    m=10,
    gen_pars={"pi": .7, "xi":.2}
)
print(mod.summary())
mod.plot()
plt.show()
```

## Manual
Work in progress.

- [Manual](./Manual) for models' specifications and working examples
- [Reference Guide](./Manual/Reference%20Guide) for details about modules' Functions and Classes

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

@Institution: Universitas Mercatorum

@Affiliation: Graduand in Statistics & Big Data (L41)

@Date:        2023-24

@Credit:      Domenico Piccolo, Rosaria Simone

@Contacts:    cub@maxpierini.it
