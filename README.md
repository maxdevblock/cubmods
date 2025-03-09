# cubmods
Statistical package: CUB models for ordinal responses.

This package is a _first Python implementation_ of statistical methods for the models of the class CUB, proposed by Professor Domenico Piccolo, 2003.

It contains inferential methods for each family of the class CUB (with or without covariates), basic graphical tools, and methods to draw random samples from given models.

It has been implemented by Massimo Pierini in 2024. It is mainly based upon the `CUB` package in `R`, maintened by Prof.ssa Rosaria Simone.

***

## Requirements

The package requires `numpy`, `pandas`, `scipy` and  `statsmodels`.

Notice that these requirements are automatically installed using `pip` (read __Installation__ section).

The package has been currently tested up to `python=3.13.2` with required packages up to

```
numpy=2.2.3
pandas=2.2.3
scipy=1.15.2
statsmodels=0.14.4
matplotlib=3.10.1
```

## Installation

The latest stable version of the package can be installed via `pip` with

```
pip install cubmods
```

Alternatively, you can install/update to the latest build directly from GitHub `main` branch, but this could be unnstable. First, you need to have `git` installed (see [Install Git](https://github.com/git-guides/install-git) for detailed instructions for Windows, macOS and Linux). After installing `git` you'll need to restart your computer (to update the PATHs) and then you can install `cubmods` from GitHub with

```
pip install -U git+https://github.com/maxdevblock/cubmods.git@main
```

All these `pip` commands, can also be run in a `Spyder 6.x` IPython console. A `conda` environment is strongly suggested.

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

Read the __Documentation__ for further details.

## Documentation
The following is a preliminary _Manual and Reference Sheet_.

- [CUBmods’s documentation](https://cubmods.readthedocs.io/en/latest/) on ReadTheDocs

## References
  - Piccolo D. (2003). On the moments of a mixture of uniform and shifted binomial random variables. Quaderni di Statistica, 5(1):85–104
  - D'Elia A. and Piccolo D. (2005). A mixture model for preferences data analysis. Computational Statistics & Data Analysis, 49(3):917–934
  - Capecchi S. and Piccolo D. (2017). Dealing with heterogeneity in ordinal responses, Quality and Quantity, 51(5), 2375--2393
  - Iannario M. and Piccolo D. (2016a). A comprehensive framework for regression models of ordinal data. Metron, 74(2), 233--252
  - Iannario M. and Piccolo D. (2016b). A generalized framework for modelling ordinal data. Statistical Methods and Applications, 25, 163--189
  - Manisera M, Zuccolotto P (2014a). Modeling “don’t know” responses in rating scales. Pattern Recognit Lett, 45:226–234
  - Piccolo D., Simone R. and Iannario M. (2019). Cumulative and CUB models for rating data: a comparative analysis. International Statistical Review, 87(2), 207-236
  - Piccolo D. and Simone R. (2019). The class of CUB models: statistical foundations, inferential issues and empirical evidence. Statistical Methods & Applications, 28, 389-435
  - Pierini M. (2024). Modelli della classe CUB in python. Bachelor's thesis L-41. Universitas Mercatorum, Rome, IT, 1–-79

## Credits
@Author:      Massimo Pierini

@Date:        2023-24

@ThanksTo:    Domenico Piccolo, Rosaria Simone

@Contacts:    cub@maxpierini.it
