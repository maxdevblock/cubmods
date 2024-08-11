Manual
======

The package ``cubmods`` can be used to apply inferential methods to an observed sample in order to 
estimate the parameters and the covariance matrix of a model within the CUB class. Also, for each family, 
random samples can be drawn from a specified model.

Currently, six families have been defined and implemented: 

- CUB (Combination of Uniform and Binomial)
- CUBSH (CUB + a SHelter choice)
- CUSH (Combination of Uniform and a SHelter choice)
- 2-CUSH (Combination of Uniform and 2 SHelter choices)
- CUBE (Combination of Uniform and BEta-binomial)
- IHG (Inverse HyperGeometric)

For each family, a model can be defined with or without covariates for one or more parameters.

Details about each family and examples are provided in the following chapters.

Even if each family has got its own *Maximum Likelihood Estimation* function ``mle()`` that 
could be called directly, for example ``cub.mle()``, the function ``gem.from_formula()`` provides a 
simplified and generalised procedure for MLE. In this manual ``gem`` will be used for the examples.

On the contrary, a general function to draw random samples has not been currently 
implemented yet and the function must be called from the module of the corresponding family, 
for example ``cube_ywz.draw()``.

The last chapter, shows the basic usage for the tool ``multicub``.

GEM syntax
----------

The function ``gem.from_formula()`` is the main function that simplifies the estimation and 
validation of a model.

A ``pandas`` DataFrame must be passed to the function, with the *kwarg* ``df=``.

The function needs a *formula* that is a **string** specifying the name of the ordinal 
variable (before the ``~`` symbol)
and of the covariates (after the symbol ``~``). Covariates for each component are
separated by the symbol ``|`` (pipeline).
The symbol ``0`` indicates no covariates for a certain component. 
If more covariates explain a single component, the symbol ``+`` concatenates the names.
Qualitative variables names, must be placed between brackets ``()`` leaded by a ``C``.

.. warning::

    No columns in the DataFrame must be named ``constant`` or ``0``.
    In the column names, are only allowed letters, numbers, and underscores ``_``.
    No space is allowed in the column names.

.. warning::

    Spaces are currently not allowed in the formula string.

If no ``model=`` *kwarg* is declared, the function takes ``"cub"`` as default.

For example, let's suppose we have a DataFrame where ``response`` is the ordinal variable, 
``age`` and ``sex`` are a quantitative and a qualitative variable to explain the *feeling* component
only in a ``cub`` family model. The formula will be ``formula = "response~0|age+C(sex)|0"``.

.. warning::

    Currently, the number of fields separated by ``|`` in a formula **MUST BE** three
    even if the specified model family has less parameters (such as ``cub`` or ``cush``).

CUB family
----------

Without covariates
^^^^^^^^^^^^^^^^^^

A model of the CUB family for responses with :math:`m` ordinal categories, without covariates is specified as

.. math::
    \Pr(R=r|\boldsymbol{\theta}) = \pi \dbinom{m-1}{r-1}(1-\xi)^{r-1}\xi^{m-r}+\dfrac{1-\pi}{m}

where :math:`\pi` and :math:`\xi`` are the parameters for respectively the *uncertainty* and the 
*feeling* components.

Note that :math:`(1-\pi)` is the weight of the Uncertainty component and 
:math:`(1-\xi)` is the Feeling component for usual *positive wording*.

In the following example, a sample will be drawn from a CUB model of :math:`n=500` observations of an ordinal 
variable with :math:`m=10` ordinal categories
and parameters :math:`(\pi=.7, \xi=.2)`. A ``seed=1`` will be set to ensure reproducibility.

.. code-block:: python
   :caption: Script
   :linenos:

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

.. code-block:: none

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

.. image:: /img/cub00draw.png
    :alt: CUB00 drawn sample


Using the previously drawn sample, in the next example the parameters :math:`(\hat\pi, \hat\xi)` will be estimated.

Note that in the function ``gem.from_formula``:

- ``df`` needs to be a ``pandas`` DataFrame; the function ``drawn.as_dataframe()`` will return a DataFrame with ``ordinal`` as default column name

- ``formula`` specifies the ordinal variable (``ordinal`` in this case) and the covariates for each component (none in this case, so ``"0|0|0"``)

- if ``m`` is not provided, the maximum observed ordinal value will be assumed

- with ``gen_pars`` dictionary, the parameters of a known model (if any) can be specified; in this case, they'll be the parameters used to draw the sample

.. code-block:: python
    :caption: Script
    :linenos:

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

.. code-block:: none

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

.. image:: /img/cub00mle.png
    :alt: CUB00 MLE

With covariates
^^^^^^^^^^^^^^^^^^

All three combinations of covariates has been implemented in both Python and R:
for *uncertainty* only, for *feeling* only, and for *both*.

Here we'll show an example with covariates for *feeling* only.

First of all, we'll draw a random sample with two covariates for the *feeling* component:
``W1`` and ``W2``. Note that, having two covariates, we'll need three :math:`\gamma` parameters,
to consider the constant term too.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods import cub_0w, gem
    # Draw a random sample
    n = 1000
    np.random.seed(1)
    W1 = np.random.randint(1, 10, n)
    np.random.seed(42)
    W2 = np.random.randint(1, 10, n)
    df = pd.DataFrame({
        "W1": W1, "W2": W2
    })
    drawn = cub_0w.draw(m=10, n=n, 
        pi=0.8,
        gamma=[2.3, -0.4, -0.05],
        W=df
    )
    drawn.plot()
    plt.show()

Then, we'll add the drawn sample to ``df`` DataFrame and will estimate the parameters.

.. code-block:: python
    :caption: Script
    :linenos:

    # add the drawn sample
    df["ordinal"] = drawn.rv
    # MLE estimation
    mod1 = gem.from_formula(
        formula="ordinal~0|W1+W2|0",
        df=df
    )
    # Print MLE summary
    print(mod1.summary())
    # plot the results
    mod1.plot()
    plt.show()



CUBSH family
------------

CUSH family
-----------

2-CUSH family
-------------

CUBE family
-----------

IHG family
----------

MULTICUB
--------
