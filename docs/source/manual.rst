Manual
======

The package ``cubmods`` can be used to apply inferential methods to an observed sample in order to 
estimate the parameters and the covariance matrix of a model within the CUB class. Also, for each family, 
random samples can be drawn from a specified model.

Currently, six families have been defined and implemented: 

- CUB (Combination of Uniform and Binomial)
- CUBSH (CUB + a SHelter choice)
- CUSH (Combination of Uniform and a SHelter choice)
- CUSH2 (Combination of Uniform and 2 SHelter choices)
- CUBE (Combination of Uniform and BEta-binomial)
- IHG (Inverse HyperGeometric)

For each family, a model can be defined with or without covariates for one or more parameters.

Details about each family and examples are provided in the following chapters.

Even if each family has got its own *Maximum Likelihood Estimation* function ``mle()`` that 
could be called directly, for example ``cub.mle()``, the function ``gem.estimate()`` provides a 
simplified and generalised procedure for MLE.

Similarly, even if each family has got its own *Random Sample Drawing* function ``draw()`` that 
could be called directly, for example ``cub.draw()``, the function ``gem.draw()`` provides a 
simplified and generalised procedure to draw a random sample.

In this manual ``gem`` functions will be used for the examples.

The last chapter, shows the basic usage for the tool ``multicub``.

GeM usage
---------

GeM (Generalized Mixture) is the main module of ``cubmods`` package, which provides simplified and
generalized functions to both estimate a model from an observed sample and draw a random sample from a 
specified model.

The function ``gem.estimate()`` is the main function for the estimation and 
validation of a model from an observed sample, calling for the corresponding ``.mle()`` function of
the specified family, with or without covariates.

The function ``gem.draw()`` is the main function for drawing a random sample from a specified model, 
calling for the corresponding ``.draw()`` function of the corresponding family,
with or without covariates.

The *formula* syntax
^^^^^^^^^^^^^^^^^^^^

Both functions need a ``formula`` that is a **string** specifying the name of the ordinal 
variable (before the tilde ``~`` symbol)
and of the covariates of the components (after the tilde symbol ``~``).
Covariates for each component are
separated by the pipeline symbol ``|``.
The *zero* symbol ``0`` indicates no covariates for a certain component. 
The *one* symbol ``1`` indicates that we want to estimate the parameter of the constant term only.
If more covariates explain a single component, the symbol ``+`` concatenates the names.
Qualitative variables names, must be placed between brackets ``()`` leaded by a ``C``,
for example ``C(varname)``.

.. warning::

    No columns in the DataFrame should be named ``constant``, ``1`` or ``0``.
    In the column names, only letters, numbers, and underscores ``_`` are allowed.
    Spaces **SHOULD NOT BE** used in the column names, but replaced with ``_``.

For example, let's suppose we have a DataFrame where ``response`` is the ordinal variable, 
``age`` and ``sex`` are a quantitative and a qualitative variable to explain the *feeling* component
only in a ``cub`` family model. The formula will be ``formula = "response ~ 0 | age + C(sex)"``.

Notice that spaces are allowed between symbols and variable names in the formula but they aren't
needed: a formula ``"ord ~ X | Y1 + Y2 | Z"`` is the same as ``"ord~X|Y1+Y2|Z"``.

.. warning::

    The number of fields separated by the pipeline ``|`` in a formula **MUST BE** equal to
    the number of parameters specifying the model family. Therefore: two for ``cub`` and ``cush2``, 
    three for ``cube`` and ``cub`` with shelter effect, one for ``cush`` and ``ihg``.

Arguments of ``estimate`` and ``draw``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within the function ``estimate`` the number of ordinal categories ``m`` is internally retrieved if not specified 
(taking the maximum observed category)
but it is advisable to pass it as an argument to the call if some category has zero frequency.
Within the function ``draw`` instead, the number of ordinal categories ``m`` must always be specified.

A ``pandas`` DataFrame must always be passed to the function ``estimate``, with the *kwarg* ``df``. 
It should contain, at least, a column of the observed sample and the columns of the covariates (if any).
If no ``df`` is passed to the function ``draw`` for a model without covariates
instead, an empty DataFrame will be created.

The number ``n`` of ordinal responses to be drawn should always be specified in the function ``draw``
for models without covariates. For model with covariates instead, ``n`` is not effective because
the number of drawn ordinal responsed will be equal to the passed DataFrame rows.

A ``seed`` could be specified for the function ``draw`` to ensure reproducibility.
Notice that, for models with covariates, ``seed`` cannot be ``0`` (in case, it will be
automatically set to ``1``).

If no ``model=`` *kwarg* is declared, the function takes ``"cub"`` as default.
Currently implemented models are: ``"cub"`` (default), ``"cush"``, ``"cube"``,
``"ihg"``, and ``"cush2"``. CUB models with shelter effect, are automatically
implemented using ``model="cub"`` and specifying a shelter choice with the 
*kwarg* ``sh``.

If  ``model="cub"`` (or nothing), then a CUB mixture model is fitted to the data to explain uncertainty, 
feeling and possible shelter effect by further passing the extra argument ``sh`` for the corresponding category.
Subjects' covariates can be included by specifying covariates matrices in the 
formula as ``ordinal~Y|W|X``,  to explain uncertainty (Y), feeling (W) or shelter (X). 
Notice that
covariates for shelter effect can be included only if specified for both feeling and uncertainty (GeCUB models). 
Nevertheless, the symbol ``1`` could be used to specify a different combination of components with covariates.
For example, if we want to specify a CUB model with covariate ``cov`` for uncertainty only, we could pass the
formula ``ordinal ~ cov | 1 | 1``: in this case, for feeling and shelter effect, the constant terms only
(:math:`\gamma_0` and :math:`\omega_0`) will be estimated and the values of the estimated :math:`\xi` and
:math:`\delta` could be computed as :math:`\hat\xi=\mathrm{expit}(\hat\gamma_0)` and 
:math:`\hat\delta=\mathrm{expit}(\hat\omega_0)`.

If ``family="cube"``, then a CUBE mixture model (Combination of Uniform and Beta-Binomial) is fitted to the data
to explain uncertainty, feeling and overdispersion.   Subjects' covariates can be also included to explain the
feeling component or all the three components by  specifying covariates matrices in the Formula as 
``ordinal~Y|W|Z`` to explain uncertainty (Y), feeling (W) or 
overdispersion (Z). For different combinations of components with covariates, the symbol ``1`` can be used.
Notice that :math:`\hat\phi=e^{-\hat\alpha_0}`.

If ``family="ihg"``, then an IHG model is fitted to the data. IHG models (Inverse Hypergeometric) are nested into
CUBE models. The parameter :math:`\theta` gives the probability of observing 
the first category and is therefore a direct measure of preference, attraction, pleasantness toward the 
investigated item. This is the reason why :math:`\theta` is customarily referred to as the 
preference parameter of the 
IHG model. Covariates for the preference parameter :math:`\theta` have to be specified 
in matrix form in the Formula as ``ordinal~V``.

If ``family="cush"``, then a CUSH model is fitted to the data (Combination of Uniform and SHelter effect).
The category corresponding to the inflation should be
passed via argument ``sh``. Covariates for the shelter parameter :math:`\delta`
are specified in matrix form Formula as ``ordinal~X``.

If ``family="cush2"``, then a 2-CUSH model is fitted to the data (Combination of Uniform and 2 SHelter choices).
The categories corresponding to the inflation should be
passed as a list (or array) via the same argument ``sh``. 
Covariates for the shelter parameters :math:`(\delta_1, \delta_2)`
are specified in matrix form Formula as ``ordinal~X1|X2``. Notice that, to specify covariates for a
single shelter choice, the formula should be ``ordinal~X1|0`` and not ``ordinal~0|X2``.

Extra arguments include the maximum 
number of iterations ``maxiter`` for the optimization algorithm, 
the required error tolerance ``tol``, and a dictionary of parameters of a known model
``gen_pars`` to be compared with the estimates.

Methods of ``estimate`` and ``draw``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For both functions, the methods ``.summary()`` and ``.plot()`` are always available calling the
main functions to print a summary and plot the results, respectively. For ``.plot()`` arguments
and options, see the ``CUBsample`` Class (for object returned by ``draw``) 
and the extended ``CUBres`` Classes of the corresponding
family (for objects returned by ``estimate``).

Calling the method ``.save(fname)`` the object can be saved on a file called ``fname.cub.sample``
(for ``draw``) or ``fname.cub.fit`` (for ``estimate``).

Saved objects can then be loaded using the function ``general.load(fname)``.

Attributes of ``estimate`` and ``draw``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUB family
----------

Basic family of the class CUB. See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        piccolo2003moments
        d2005mixture
        piccolo2006observed
        iannario2014inference
        piccolo2019class

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

All three combinations of covariates has been implemented for CUB family in both Python and R:
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

.. image:: /img/cub0wdraw.png
    :alt: CUB0W drawn sample

Then, we'll add the drawn sample to ``df`` DataFrame and will estimate the parameters.

.. code-block:: python
    :caption: Script
    :linenos:

    # add the drawn sample
    df["ordinal"] = drawn.rv
    # MLE estimation
    mod1 = gem.from_formula(
        formula="ordinal ~ 0 | W1+W2 | 0",
        df=df
    )
    # Print MLE summary
    print(mod1.summary())
    # plot the results
    mod1.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUB(0W) model <<<===== ML-estimates
    =======================================================================
    m=10  Size=1000  Iterations=22  Maxiter=500  Tol=1E-04
    -----------------------------------------------------------------------
    Uncertainty
            Estimates  StdErr     Wald  p-value
    pi            0.789  0.0231   34.210   0.0000
    -----------------------------------------------------------------------
    Feeling
            Estimates  StdErr     Wald  p-value
    constant      2.299  0.1001   22.976   0.0000
    W1           -0.407  0.0139  -29.239   0.0000
    W2           -0.044  0.0121   -3.681   0.0002
    =======================================================================
    Dissimilarity = 0.0488
    Loglik(MOD)   = -1963.868
    Loglik(uni)   = -2302.585
    Mean-loglik   = -1.964
    -----------------------------------------------------------------------
    AIC = 3935.74
    BIC = 3955.37
    =======================================================================
    Elapsed time=0.10014 seconds =====>>> Sun Aug 11 22:02:15 2024
    =======================================================================

.. image:: /img/cub0wmle.png
    :alt: CUB0W MLE

CUBSH family
------------

Basic family of the class CUB. See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        iannario2010new
        iannario2012modelling
        iannario2014inference
        piccolo2019class

Without covariates
^^^^^^^^^^^^^^^^^^

A model of the CUB family with shelter effect
for responses with :math:`m` ordinal categories, without covariates is specified as

.. math::
    \Pr(R=r|\boldsymbol{\theta}) = \delta D_r^{(c)} + (1-\delta)\left(\pi b_r(\xi) + \frac{1-\pi}{m} \right)

where :math:`\pi` and :math:`\xi`` are the parameters for respectively the *uncertainty* and the 
*feeling* components, and :math:`\delta` is the weight of the shelter effect.

With covariates
^^^^^^^^^^^^^^^

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
