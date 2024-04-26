The package `cubmods` can be used to apply inferential methods to an observed sample in order to estimate the parameters and covariance matrix of a model within the CUB class. Also, for each family, random samples can be drawn from a specified model.

Currently, six families have been defined and implemented: 

- CUB (Combination of Uniform and Binomial)
- CUBSH (CUB + a SHelter choice)
- CUBE (Combination of Uniform and BEta-binomial)
- IHG (Inverse HyperGeometric)
- CUSH (Combination of Uniform and a SHelter choice)
- 2-CUSH (Combination of Uniform and 2 SHelter choices)

For each family, can be defined a model with or without covariates for one or more parameters.

Details about each family and examples are provided in the following chapters.

Even if each family has got its own _Maximum Likelihood Estimation_ function `mle()` that could be called directly, for example `cub.mle()`, the function `gem.from_formula()` provides a simplified and generalised procedure for MLE. In this manual `gem` will be used for the examples.

On the contrary, a general function to draw random sample has not been currently implemented yet and the function must be called from the module of the corresponding family, for example `cube_ywz.draw()`.

The last chapter, shows the basic usage for the tool `multicub`.

# `gem` basic syntax

TODO: gem basic syntax

# Saving and loading objects

TODO: loading and saving objects

# Summary

- [CUB family](Manual/02_cub_family.md)
- [CUBSH family](Manual/03_cubsh_family.md)
- [CUBE family](Manual/04_cube_family.md)
- [IHG family](Manual/06_ihg_family.md)
- [CUSH family](Manual/05_cush_family.md)
- [2-CUSH family](Manual/07_2cush_family.md)
- [MULTICUB tool](Manual/08_multicub.md)
