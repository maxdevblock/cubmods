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

If no ``model`` is declared, the function takes ``"cub"`` as default.
Currently implemented models are: ``"cub"`` (default), ``"cush"``, ``"cube"``,
and ``"ihg"``. CUB models with shelter effect are automatically
implemented using ``model="cub"`` and specifying a shelter choice with the 
*kwarg* ``sh``. CUSH2 models are automatically
implemented using ``model="cush"`` and passing a list of two categories to
the *kwarg* ``sh`` instead of an integer.

To ``draw`` must be passed the parameters' values with the *kwargs* of the corresponding
family: for example, ``pi`` and ``xi`` for CUB models without covariates, ``beta`` and ``gamma``
for CUB models with covariates for both feeling and uncertainty, etc. See the
``.draw()`` function reference of the corresponding family module for details.

If  ``model="cub"`` (or nothing), then a CUB mixture model is fitted to the data to explain uncertainty, 
feeling (``ordinal~Y|W``) and possible shelter effect by further passing the extra argument ``sh`` for the corresponding category.
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
Notice that :math:`\hat\phi=e^{\hat\alpha_0}`.

If ``family="ihg"``, then an IHG model is fitted to the data. IHG models (Inverse Hypergeometric) are nested into
CUBE models. The parameter :math:`\theta` gives the probability of observing 
the first category and is therefore a direct measure of preference, attraction, pleasantness toward the 
investigated item. This is the reason why :math:`\theta` is customarily referred to as the 
preference parameter of the 
IHG model. Covariates for the preference parameter :math:`\theta` have to be specified 
in matrix form in the Formula as ``ordinal~V``.

If ``family="cush"``, then a CUSH model is fitted to the data (Combination of Uniform and SHelter effect).
If a category corresponding to the inflation should be
passed via argument ``sh`` a CUSH model is called and
covariates for the shelter parameter :math:`\delta`
are specified in matrix form Formula as ``ordinal~X``.
If two category corresponding to the inflation should be
passed via argument ``sh`` (as a *list* or *array*) a CUSH2 model is called and
covariates for the shelters' parameters :math:`(\delta_1,\delta_2)`
are specified in matrix form Formula as ``ordinal~X1|X2``.
Notice that, to specify covariates for a
single shelter choice in a CUSH2 model, 
the formula should be ``ordinal~X1|0`` and not ``ordinal~0|X2``.

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

Calling ``.as_dataframe()`` will return a DataFrame of parameters' names and values for objects
of the Class ``CUBsample`` returned by ``draw``. For objects of the Base Class ``CUBres`` returned
by ``estimate`` instead, will return a DataFrame with parameters' component, name, estimated value,
standard error, Wald test statistics and p-value.

Calling the method ``.save(fname)`` the object can be saved on a file called ``fname.cub.sample``
(for ``draw``) or ``fname.cub.fit`` (for ``estimate``).

Saved objects can then be loaded using the function ``general.load(fname)``.

Attributes of ``estimate`` and ``draw``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For both objects returned by ``estimate`` and ``draw``, the attributes ``.formula`` and
``.df`` are always available. The function ``draw`` will return the original DataFrame (if provided)
with an extra column of the drawn ordinal response called as specified in the formula.

Many other attributes can be called from objects of the Base Class ``CUBres`` returned by
``estimate``, such as the computed loglikelihood, the AIC and BIC, ectcetera. For details,
see the Base Class ``CUBres`` reference guide.

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
        iannario2010new
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

.. _cub-models-without-covariates:

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
    from cubmods.gem import draw

    # draw a sample
    drawn = draw(
        formula="ord ~ 0 | 0",
        m=10, pi=.7, xi=.2,
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
    formula: ord~0|0
    -----------------------------------------------------------------------
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

Notice that, since the default value of the kwarg ``model`` is
``"cub"`` we do not need to specify it.

Calling ``drawn.as_dataframe()`` will return a DataFrame with
the parameters

.. code-block:: none

      parameter  value
    0        pi    0.7
    1        xi    0.2

Using the previously drawn sample, in the next example the parameters :math:`(\hat\pi, \hat\xi)` will be estimated.

Note that in the function ``gem.estimate``:

- ``df`` needs to be a ``pandas`` DataFrame; the attribute ``drawn.df`` will return a DataFrame with ``ord`` as column name of the drawn ordinal response (as previuosly speficied in the formula)

- ``formula`` needs the ordinal variable name (``ord`` in this case) and the covariates for each component (none in this case, so ``"0|0"``)

- if ``m`` is not provided, the maximum observed ordinal value will be assumed and a warning will be raised

- with ``gen_pars`` dictionary, the parameters of a known model (if any) can be specified; in this case, we'll specify the known parameters used to draw the sample

.. code-block:: python
    :caption: Script
    :linenos:

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df,
        formula="ord~0|0",
        gen_pars={
            "pi": drawn.pars[0],
            "xi": drawn.pars[1]
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")

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

Calling ``fit.as_dataframe()`` will return a DataFrame with
parameters' estimated values and standard errors

.. code-block:: none

         component parameter  estimate    stderr       wald        pvalue
    0  Uncertainty        pi   0.67476  0.033954  19.872485  7.042905e-88
    1      Feeling        xi   0.18817  0.009043  20.807551  3.697579e-96

.. _cub-models-with-covariates:

With covariates
^^^^^^^^^^^^^^^^^^

.. math::
    \Pr(R_i=r|\pmb\theta, \pmb y_i, \pmb w_i) = \pi_i \dbinom{m-1}{r-1}(1-\xi_i)^{r-1}\xi_i^{m-r}+\dfrac{1-\pi_i}{m}

.. math::
    \left\{
    \begin{array}{l}
        \pi_i = \dfrac{1}{1+\exp\{-\pmb y_i \pmb \beta\}}
        \\
        \xi_i = \dfrac{1}{1+\exp\{-\pmb w_i \pmb \gamma\}}
    \end{array}
    \right.

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
    from cubmods.gem import draw, estimate

    # Draw a random sample
    n = 1000
    np.random.seed(1)
    W1 = np.random.randint(1, 10, n)
    np.random.seed(42)
    W2 = np.random.random(n)
    df = pd.DataFrame({
        "W1": W1, "W2": W2
    })
    drawn = draw(
        formula="res ~ 0 | W1 + W2",
        df=df,
        m=10, n=n,
        pi=0.8,
        gamma=[2.3, 0.2, -5],
    )
    # print the summary
    print(drawn.summary())

.. code-block:: none

    =======================================================================
    =====>>> CUB(0W) model <<<===== Drawn random sample
    =======================================================================
    m=10  Sample size=1000  seed=None
    formula: res~0|W1+W2
    -----------------------------------------------------------------------
    pi=0.800
    constant=2.300
    W1=0.200
    W2=-5.000
    =======================================================================
    Sample metrics
    Mean     = 4.566000
    Variance = 8.089734
    Std.Dev. = 2.844246
    -----------------------------------------------------------------------
    Dissimilarity = 0.0307673
    =======================================================================

.. code-block:: python
    :caption: Script
    :linenos:

    # plot the drawn sample
    drawn.plot()
    plt.show()

.. image:: /img/cub0wdraw.png
    :alt: CUB0W drawn sample

.. code-block:: python
    :caption: Script
    :linenos:

    # print the parameters' values
    print(drawn.as_dataframe())

.. code-block:: none

      parameter  value
    0        pi    0.8
    1  constant    2.3
    2        W1    0.2
    3        W2   -5.0

.. code-block:: python
    :caption: Script
    :linenos:

    # print the updated DataFrame
    print(drawn.df)

.. code-block:: none

         W1        W2  res
    0     6  0.374540    2
    1     9  0.950714    7
    2     6  0.731994    8
    3     1  0.598658    8
    4     1  0.156019    4
    ..   ..       ...  ...
    995   3  0.091582    2
    996   9  0.917314    9
    997   4  0.136819    1
    998   7  0.950237    3
    999   8  0.446006    2

    [1000 rows x 3 columns]

Finally, we'll call ``estimate`` to estimate the parameters
given the observed (actually, drawn) sample.

.. code-block:: python
    :caption: Script
    :linenos:

    # MLE estimation
    fit = estimate(
        formula="res ~ 0 | W1+W2",
        df=drawn.df,
    )
    # Print MLE summary
    print(fit.summary())
    # plot the results
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUB(0W) model <<<===== ML-estimates
    =======================================================================
    m=10  Size=1000  Iterations=18  Maxiter=500  Tol=1E-04
    -----------------------------------------------------------------------
    Uncertainty
              Estimates  StdErr     Wald  p-value
    pi            0.800  0.0198   40.499   0.0000
    -----------------------------------------------------------------------
    Feeling
              Estimates  StdErr     Wald  p-value
    constant      2.353  0.1001   23.514   0.0000
    W1            0.194  0.0138   14.034   0.0000
    W2           -5.076  0.1454  -34.909   0.0000
    =======================================================================
    Dissimilarity = 0.0292
    Loglik(MOD)   = -1807.052
    Loglik(uni)   = -2302.585
    Mean-loglik   = -1.807
    -----------------------------------------------------------------------
    AIC = 3622.10
    BIC = 3641.74
    =======================================================================
    Elapsed time=0.09656 seconds =====>>> Thu Aug 15 18:31:21 2024
    =======================================================================

.. image:: /img/cub0wmle.png
    :alt: CUB0W MLE

CUBSH family
------------

Basic family of the class CUB with shelter effect. 
See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        iannario2012modelling
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

.. _cubsh-without-covariates:

Without covariates
^^^^^^^^^^^^^^^^^^

A model of the CUB family with shelter effect
for responses with :math:`m` ordinal categories, without covariates is specified as

.. math::
    \Pr(R=r|\boldsymbol{\theta}) = \delta D_r^{(c)} + (1-\delta)\left(\pi b_r(\xi) + \frac{1-\pi}{m} \right)

where :math:`\pi` and :math:`\xi` are the parameters for respectively the *uncertainty* and the 
*feeling* components, and :math:`\delta` is the weight of the shelter effect.

In the next example, we'll draw an ordinal response
and then estimate the parameters given the sample.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import matplotlib.pyplot as plt
    from cubmods.gem import draw, estimate

    # draw a sample
    drawn = draw(
        formula="ord ~ 0 | 0 | 0",
        m=7, sh=1,
        pi=.8, xi=.4, delta=.15,
        n=1500, seed=42)

    print(drawn.as_dataframe())

.. code-block:: none

      parameter  value
    0       pi1   0.68
    1       pi2   0.17
    2        xi   0.40
    3       *pi   0.80
    4    *delta   0.15

Notice that:

- since ``"cub"`` is default value of the *kwarg* ``model``, we do not need to specify it

- we'll pass to ``estimate`` *kwargs* values taken from the object ``drawn``

.. code-block:: python
    :caption: Script
    :linenos:

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df, sh=drawn.sh,
        formula=drawn.formula,
        gen_pars={
            "pi1": drawn.pars[0],
            "pi2": drawn.pars[1],
            "xi": drawn.pars[2],
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUBSH model <<<===== ML-estimates
    =======================================================================
    m=7  Shelter=1  Size=1500  Iterations=59  Maxiter=500  Tol=1E-04
    -----------------------------------------------------------------------
    Alternative parametrization
           Estimates  StdErr    Wald  p-value
    pi1        0.661  0.0307  21.508   0.0000
    pi2        0.174  0.0344   5.041   0.0000
    xi         0.388  0.0077  50.592   0.0000
    -----------------------------------------------------------------------
    Uncertainty
           Estimates  StdErr    Wald  p-value
    pi         0.792  0.0400  19.813   0.0000
    -----------------------------------------------------------------------
    Feeling
           Estimates  StdErr    Wald  p-value
    xi         0.388  0.0077  50.592   0.0000
    -----------------------------------------------------------------------
    Shelter effect
           Estimates  StdErr    Wald  p-value
    delta      0.166  0.0116  14.327   0.0000
    =======================================================================
    Dissimilarity = 0.0049
    Loglik(sat)   = -2734.302
    Loglik(MOD)   = -2734.433
    Loglik(uni)   = -2918.865
    Mean-loglik   = -1.823
    Deviance      = 0.263
    -----------------------------------------------------------------------
    AIC = 5474.87
    BIC = 5490.81
    =======================================================================

.. image:: /img/cubsh00mle.png
    :alt: CUBSH 00 MLE

.. _cubsh-with-covariates:

With covariates
^^^^^^^^^^^^^^^

.. math::
    \Pr(R_i=r|\pmb\theta, \pmb y_i, \pmb w_i, \pmb x_i) = \delta_i D_r^{(c)} + (1-\delta_i)\left(\pi_i b_r(\xi_i) + \frac{1-\pi_i}{m} \right)

.. math::
    \left\{
    \begin{array}{l}
        \pi_i = \dfrac{1}{1+\exp\{-\pmb y_i \pmb \beta\}}
        \\
        \xi_i = \dfrac{1}{1+\exp\{-\pmb w_i \pmb \gamma\}}
        \\
        \delta_i = \dfrac{1}{1+\exp\{-\pmb x_i \pmb \omega\}}
    \end{array}
    \right.

Only the model with covariates for all components has been
currently defined and implemented.

Nevertheless, thanks to the symbol ``1`` provided by the
*formula*, we can specify a different combination
of covariates.

For example, we'll specifiy a model CUB with shelter effect,
with covariates for uncertainty only. We'll use the function
``logit`` to have better 'control' of the parameters values,
because :math:`\gamma_0 = \mathrm{logit}(\xi)` and
similarly for :math:`\pi` and :math:`\delta`.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods.general import expit, logit
    from cubmods.gem import draw, estimate

    # Draw a random sample
    n = 1000
    np.random.seed(1)
    W1 = np.random.randint(1, 10, n)
    df = pd.DataFrame({
        "W1": W1,
    })
    drawn = draw(
        formula="fee ~ W1 | 1 | 1",
        df=df,
        m=9, sh=2,
        beta=[logit(.8), -.2],
        gamma=[logit(.3)],
        omega=[logit(.12)],
    )

    # MLE estimation
    fit = estimate(
        formula="fee ~ W1 | 1 | 1",
        df=drawn.df, sh=2,
    )
    # Print MLE summary
    print(fit.summary())
    # plot the results
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUBSH(YWX) model <<<===== ML-estimates
    =======================================================================
    m=9  Shelter=2  Size=1000  Iterations=25  Maxiter=500  Tol=1E-04
    -----------------------------------------------------------------------
    Uncertainty
              Estimates  StdErr     Wald  p-value
    constant      0.992  0.3314    2.994   0.0028
    W1           -0.127  0.0569   -2.228   0.0259
    -----------------------------------------------------------------------
    Feeling
              Estimates  StdErr     Wald  p-value
    constant     -0.902  0.0381  -23.662   0.0000
    -----------------------------------------------------------------------
    Shelter effect
              Estimates  StdErr     Wald  p-value
    constant     -2.074  0.1260  -16.462   0.0000
    =======================================================================
    Dissimilarity = 0.0139
    Loglik(MOD)   = -2069.978
    Loglik(uni)   = -2197.225
    Mean-loglik   = -2.070
    -----------------------------------------------------------------------
    AIC = 4147.96
    BIC = 4167.59
    =======================================================================
    Elapsed time=1.43850 seconds =====>>> Thu Aug 15 19:39:49 2024
    =======================================================================

.. image:: /img/cubshywxmle.png
    :alt: CUBSH YWX MLE

To get the estimated values of :math:`\hat\xi` and :math:`\hat\delta`
we can use the function ``expit`` because :math:`\hat\xi = \mathrm{expit}(\hat\gamma_0)`
and similarly for :math:`\hat\delta`. Then, since
:math:`\widehat{es}(\xi) = \mathrm{expit}[\hat\gamma_0+\widehat{es}(\gamma_0)] - \hat\xi`
we can compute the standard errors of both :math:`\hat\xi` and :math:`\hat\delta`.

.. code-block:: python
    :caption: Script
    :linenos:

    est_xi = expit(fit.estimates[2])
    est_de = expit(fit.estimates[3])
    est_xi_se = expit(fit.estimates[2]+fit.stderrs[2]) - est_xi
    est_de_se = expit(fit.estimates[3]+fit.stderrs[3]) - est_de
    print(
        "     estimates  stderr\n"
        f"xi      {est_xi:.4f}  {est_xi_se:.4f}"
        "\n"
        f"delta   {est_de:.4f}  {est_de_se:.4f}"
    )

.. code-block:: none

         estimates  stderr
    xi      0.2886  0.0079
    delta   0.1116  0.0131

which, in fact, match the values used to draw the sample.

CUSH family
-----------

Basic family of the class CUSH with a single shelter effect. 
See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        capecchi2017dealing
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

.. _cush-without-covariates:

Without covariates
^^^^^^^^^^^^^^^^^^

.. math::
    \Pr(R=r|\pmb\theta) = \delta D_r^{(c)} + (1-\delta)/m

In the example, we'll draw a sample from a CUSH model without covariates and
then estimate the parameter given the observed sample.

Notice that, since the ``model`` is not the default ``"cub"``, we need to specify it.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import matplotlib.pyplot as plt
    from cubmods.gem import draw, estimate

    # draw a sample
    drawn = draw(
        formula="ord ~ 0",
        model="cush",
        sh=7,
        m=7, delta=.15,
        n=1500, seed=76)

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df,
        model="cush",
        formula="ord~0",
        sh=7,
        gen_pars={
            "delta": drawn.pars[0],
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUSH model <<<===== ML-estimates
    =======================================================================
    m=7  Shelter=7  Size=1500  
    -----------------------------------------------------------------------
    Shelter effect
           Estimates  StdErr   Wald  p-value
    delta      0.124  0.0130  9.532   0.0000
    =======================================================================
    Dissimilarity = 0.0236
    Loglik(sat)   = -2856.039
    Loglik(MOD)   = -2859.923
    Loglik(uni)   = -2918.865
    Mean-loglik   = -1.907
    Deviance      = 7.768
    -----------------------------------------------------------------------
    AIC = 5721.85
    BIC = 5727.16
    =======================================================================
    Elapsed time=0.00113 seconds =====>>> Fri Aug 16 10:44:07 2024
    =======================================================================

.. image:: /img/cush0mle.png
    :alt: CUSH 0 MLE

.. _cush-with-covariates:

With covariates
^^^^^^^^^^^^^^^

.. math::
    \Pr(R_i=r|\pmb\theta,\pmb x_i) = \delta_i D_r^{(c)} + (1-\delta_i)/m

.. math::
    \delta_i = \dfrac{1}{1+\exp\{ - \pmb x_i \pmb\omega \}}

In the example, we'll draw a sample from a CUSH model with covariates and
then estimate the parameter given the observed sample.

Notice that, since the ``model`` is not the default ``"cub"``, we need to specify it.

.. code-block:: python
    :caption: Script

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods.general import logit
    from cubmods.gem import draw, estimate

    # Draw a random sample
    n = 1000
    np.random.seed(1)
    X = np.random.randint(1, 10, n)
    df = pd.DataFrame({
        "X": X,
    })
    drawn = draw(
        formula="fee ~ X",
        model="cush",
        df=df,
        m=9, sh=5,
        omega=[logit(.05), .2],
    )

    # MLE estimation
    fit = estimate(
        formula="fee ~ X",
        model="cush",
        df=drawn.df, sh=5,
    )
    # Print MLE summary
    print(fit.summary())
    # plot the results
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUSH(X) model <<<===== ML-estimates
    =======================================================================
    m=9  Shelter=5  Size=1000  
    -----------------------------------------------------------------------
    Shelter effect
              Estimates  StdErr    Wald  p-value
    constant     -3.131  0.4361  -7.180   0.0000
    X             0.229  0.0629   3.642   0.0003
    =======================================================================
    Dissimilarity = 0.0395
    Loglik(MOD)   = -2130.030
    Loglik(uni)   = -2197.225
    Mean-loglik   = -2.130
    -----------------------------------------------------------------------
    AIC = 4264.06
    BIC = 4273.87
    =======================================================================
    Elapsed time=0.01704 seconds =====>>> Fri Aug 16 10:54:11 2024
    =======================================================================

.. image:: /img/cushxmle.png
    :alt: CUSH X MLE

CUSH2 family
-------------

Family of the class CUSH with two shelter effects (CUSH2). 
See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        mythesis
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

.. _cush2-without-covariates:

Without covariates
^^^^^^^^^^^^^^^^^^

.. math::
    \Pr(R=r|\pmb\theta) = \delta_1 D_r^{(c_1)} + \delta_2 D_r^{(c_2)} + (1-\delta_1-\delta_2)/m

In the example, we'll draw a sample from a CUSH2 model without covariates and
then estimate the parameter given the observed sample.

Notice that, since the ``model`` is not the default ``"cub"``, we need to specify it.
Passing a list of two shelter categories with the *kwarg* ``sh``, a CUSH2 model will be
called.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import matplotlib.pyplot as plt
    from cubmods.gem import draw, estimate

    # draw a sample
    drawn = draw(
        formula="ord ~ 0 | 0",
        model="cush",
        sh=[1,7],
        m=7,
        delta1=.15, delta2=.1,
        n=1000, seed=42)

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df,
        model="cush",
        formula="ord~0|0",
        sh=[1,7],
        gen_pars={
            "delta1": drawn.pars[0],
            "delta2": drawn.pars[1],
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUSH2 model <<<===== ML-estimates
    =======================================================================
    m=7  Shelter=[1 7]  Size=1000  
    -----------------------------------------------------------------------
    Shelter effects
            Estimates  StdErr    Wald  p-value
    delta1      0.171  0.0148  11.535   0.0000
    delta2      0.107  0.0163   6.555   0.0000
    =======================================================================
    Dissimilarity = 0.0184
    Loglik(sat)   = -1852.818
    Loglik(MOD)   = -1854.344
    Loglik(uni)   = -1945.910
    Mean-loglik   = -1.854
    Deviance      = 3.053
    -----------------------------------------------------------------------
    AIC = 3712.69
    BIC = 3722.50
    =======================================================================
    Elapsed time=0.00228 seconds =====>>> Fri Aug 16 11:15:21 2024
    =======================================================================

.. image:: /img/cush200mle.png
    :alt: CUSH2 00 MLE

.. _cush2-with-covariates:

With covariates
^^^^^^^^^^^^^^^

.. math::
    \Pr(R_i=r|\pmb\theta,\pmb x_{1i}, \pmb x_{2i}) = \delta_{1i} D_r^{(c_1)} + \delta_{2i} D_r^{(c_2)} + (1-\delta_{1i}- \delta_{2i})/m

.. math::
    \left\{
    \begin{array}{l}
        \delta_{1i} = \dfrac{1}{1+\exp\{ - \pmb x_{1i} \pmb\omega_1 \}}
        \\
        \delta_{2i} = \dfrac{1}{1+\exp\{ - \pmb x_{2i} \pmb\omega_2 \}}
    \end{array}
    \right.

Two CUSH2 models with covariates have been defined and implemented:
for the first shelter choice only and for both.

In this example we'll draw a sample from a CUSH2 model with
covariates for the first shelter choice only and will then
estimate the parameters with a CUSH2 model with covariates
for both shelter choices but using the symbol ``1`` in the
formula for the second shelter choice to estimate the
constant parameter only. This is usually not needed, but
we do it here to confirm that :math:`\mathrm{expit}(\hat\omega_{20})=\hat\delta_2`.

Notice that, since the ``model`` is not the default ``"cub"``, we need to specify it.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods.general import logit, expit
    from cubmods.gem import draw, estimate

    # Draw a random sample
    n = 1000
    np.random.seed(1)
    X = np.random.randint(1, 10, n)
    df = pd.DataFrame({
        "X": X,
    })
    drawn = draw(
        formula="fee ~ X | 0",
        model="cush",
        df=df,
        m=9, sh=[2, 8],
        omega1=[logit(.05), .2],
        delta2=.1
    )

    # MLE estimation
    fit = estimate(
        formula="fee ~ X | 1",
        model="cush",
        df=drawn.df, sh=[2, 8],
    )
    # Print MLE summary
    print(fit.summary())
    # plot the results
    fit.plot()
    plt.show()

    est_de2 = expit(fit.estimates[2])
    est_de2_es = expit(fit.estimates[2]+fit.stderrs[2]) - est_de2
    print(
        "     estimates  stderr\n"
        f"delta2  {est_de2:.4f}  {est_de2_es:.4f}"
    )

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUSH2(X1,X2) model <<<===== ML-estimates
    =======================================================================
    m=9  Shelter=[2 8]  Size=1000  
    -----------------------------------------------------------------------
    Shelter effect 1
              Estimates  StdErr     Wald  p-value
    constant     -3.170  0.4216   -7.519   0.0000
    X             0.207  0.0613    3.379   0.0007
    -----------------------------------------------------------------------
    Shelter effect 2
              Estimates  StdErr     Wald  p-value
    constant     -2.276  0.1609  -14.149   0.0000
    =======================================================================
    Dissimilarity = 0.0305
    Loglik(MOD)   = -2122.463
    Loglik(uni)   = -2197.225
    Mean-loglik   = -2.122
    -----------------------------------------------------------------------
    AIC = 4250.93
    BIC = 4265.65
    =======================================================================
    Elapsed time=0.06553 seconds =====>>> Fri Aug 16 11:29:11 2024
    =======================================================================

.. image:: /img/cush2xxmle.png
    :alt: CUSH2 XX MLE

.. code-block:: none

         estimates  stderr
    delta2  0.0931  0.0145

CUBE family
-----------

Family of the class CUBE (Combination of Uniform and BEtaBinomial). 
See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        iannario2014modelling
        iannario2015detecting
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

.. _cube-without-covariates:

Without covariates
^^^^^^^^^^^^^^^^^^

.. math::
    \Pr(R=r|\pmb{\theta}) = \pi \beta e(\xi,\phi)+\dfrac{1-\pi}{m}

In this example, we'll draw a sample from a CUBE model and then
will estimate the parameters given the observed sample.

Notice that, since the ``model`` is not the default ``"cub"``, we need to specify it.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import matplotlib.pyplot as plt
    from cubmods.gem import draw, estimate

    # draw a sample
    drawn = draw(
        formula="ord ~ 0 | 0 | 0",
        model="cube",
        m=9, pi=.7, xi=.3, phi=.15,
        n=500, seed=1)

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df,
        formula="ord~0|0|0",
        model="cube",
        gen_pars={
            "pi": drawn.pars[0],
            "xi": drawn.pars[1],
            "phi": drawn.pars[2],
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUBE model <<<===== ML-estimates
    =======================================================================
    m=9  Size=500  Iterations=62  Maxiter=1000  Tol=1E-06
    -----------------------------------------------------------------------
    Uncertainty
         Estimates  StdErr    Wald  p-value
    pi       0.577  0.0633   9.108   0.0000
    -----------------------------------------------------------------------
    Feeling
         Estimates  StdErr    Wald  p-value
    xi       0.251  0.0217  11.560   0.0000
    -----------------------------------------------------------------------
    Overdispersion
         Estimates  StdErr    Wald  p-value
    phi      0.111  0.0402   2.754   0.0059
    =======================================================================
    Dissimilarity = 0.0426
    Loglik(sat)   = -1037.855
    Loglik(MOD)   = -1041.100
    Loglik(uni)   = -1098.612
    Mean-loglik   = -2.082
    Deviance      = 6.491
    -----------------------------------------------------------------------
    AIC = 2088.20
    BIC = 2100.84
    =======================================================================
    Elapsed time=0.07919 seconds =====>>> Fri Aug 16 12:18:49 2024
    =======================================================================

.. image:: /img/cube000mle.png
    :alt: CUBE 000 MLE

.. _cube-with-covariates:

With covariates
^^^^^^^^^^^^^^^

.. math::
    \Pr(R_i=r|\pmb{\theta};\pmb y_i, \pmb w_i; \pmb z_i) = \pi_i \beta e(\xi_i,\phi_i)+\dfrac{1-\pi_i}{m}

.. math::
    \left\{
    \begin{array}{l}
        \pi_i = \dfrac{1}{1+\exp\{ -\pmb y_i \pmb\beta\}}
        \\
        \xi_i = \dfrac{1}{1+\exp\{ -\pmb w_i \pmb\gamma\}}
        \\
        \phi_i = \exp\{ \pmb z_i \pmb \alpha \}
    \end{array}
    \right.

Currently, two CUBE models have been defined and implemented:
for the *feeling* only and for all components.
Nevertheless, the symbol ``1`` can always be used in the
formula for different combinations of covariates.

In this example, we'll draw a sample with covariates for
*feeling* only and then will estimate the parameters given
the observed sample.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods.general import expit, logit
    from cubmods.gem import draw, estimate

    # Draw a random sample
    n = 1000
    np.random.seed(76)
    W = np.random.randint(1, 10, n)
    df = pd.DataFrame({
        "W": W,
    })
    drawn = draw(
        formula="fee ~ 0 | W | 0",
        model="cube",
        df=df,
        m=9,
        pi=.8,
        gamma=[logit(.3), -.1],
        phi=.12,
    )

    # MLE estimation
    fit = estimate(
        formula="fee ~ 0 | W | 0",
        model="cube",
        df=drawn.df,
    )
    # Print MLE summary
    print(fit.summary())
    # plot the results
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUBE(0W0) model <<<===== ML-estimates
    =======================================================================
    m=9  Size=1000  
    -----------------------------------------------------------------------
    Uncertainty
              Estimates  StdErr    Wald  p-value
    pi            0.815  0.0343  23.733   0.0000
    -----------------------------------------------------------------------
    Feeling
              Estimates  StdErr    Wald  p-value
    constant     -0.770  0.1012  -7.612   0.0000
    W            -0.116  0.0191  -6.052   0.0000
    -----------------------------------------------------------------------
    Overdisperson
              Estimates  StdErr    Wald  p-value
    phi           0.150  0.0260   5.779   0.0000
    =======================================================================
    Dissimilarity = 0.0183
    Loglik(MOD)   = -1886.654
    Loglik(uni)   = -2197.225
    Mean-loglik   = -1.887
    -----------------------------------------------------------------------
    AIC = 3781.31
    BIC = 3800.94
    =======================================================================
    Elapsed time=2.30903 seconds =====>>> Fri Aug 16 12:31:10 2024
    =======================================================================

.. image:: /img/cube0w0mle.png
    :alt: CUBE 0W0 MLE

Notice that the same results can be achieved using a CUBE
model with covariates for all components and passing
the symbol ``1`` to the *uncertainty* and *overdispersion*
components.

.. code-block:: python
    :caption: Script
    :linenos:

    # MLE estimation
    fit = estimate(
        formula="fee ~ 1 | W | 1",
        model="cube",
        df=drawn.df,
    )
    # Print MLE summary
    print(fit.summary())
    # plot the results
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> CUBE(YWZ) model <<<===== ML-estimates
    =======================================================================
    m=9  Size=1000  Iterations=29  Maxiter=1000  Tol=1E-02
    -----------------------------------------------------------------------
    Uncertainty
              Estimates  StdErr     Wald  p-value
    constant      1.423  0.2183    6.518   0.0000
    -----------------------------------------------------------------------
    Feeling
              Estimates  StdErr     Wald  p-value
    constant     -0.778  0.1018   -7.639   0.0000
    W            -0.117  0.0193   -6.074   0.0000
    -----------------------------------------------------------------------
    Overdispersion
              Estimates  StdErr     Wald  p-value
    constant     -1.930  0.1756  -10.989   0.0000
    =======================================================================
    Dissimilarity = 0.0239
    Loglik(MOD)   = -1886.690
    Loglik(uni)   = -2197.225
    Mean-loglik   = -1.887
    -----------------------------------------------------------------------
    AIC = 3781.38
    BIC = 3801.01
    =======================================================================
    Elapsed time=50.02969 seconds =====>>> Fri Aug 16 12:33:36 2024
    =======================================================================

.. image:: /img/cubeywzmle.png
    :alt: CUBE YWZ MLE

In fact:

.. code-block:: python
    :caption: Script
    :linenos:

    est_pi = expit(fit.estimates[0])
    est_ph = np.exp(fit.estimates[3])
    est_pi_se = expit(fit.estimates[0]+fit.stderrs[0]) - est_pi
    est_ph_se = np.exp(fit.estimates[3]+fit.stderrs[3]) - est_ph
    print(
        "     estimates  stderr\n"
        f"pi      {est_pi:.4f}  {est_pi_se:.4f}"
        "\n"
        f"phi     {est_ph:.4f}  {est_ph_se:.4f}"
    )

.. code-block:: none

         estimates  stderr
    pi      0.8058  0.0319
    phi     0.1451  0.0279

IHG family
----------

Family of the class IHG (Inverse Hyper Geometric). 
See the references for details.

References
^^^^^^^^^^

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        d2003modelling
        d2005moment
        iannario2009program
        iannario2014inference
        iannario2022package
        piccolo2019class

.. _ihg-without-covariates:

Without covariates
^^^^^^^^^^^^^^^^^^

.. math::
    \left\{
    \begin{array}{l}
        \Pr(R=1|\theta) = \theta
        \\
        \Pr(R=r+1|\theta) = \Pr(R=r)(1-\theta)\dfrac{m-r}{m-1-r(1-\theta)},\; r= 1, \ldots, m-1
    \end{array}
    \right.

In this example, we'll draw a sample from an IHG model
and the estimate the parameter from the observed sample.

.. code-block:: python

    # import libraries
    import matplotlib.pyplot as plt
    from cubmods.gem import draw, estimate

    # draw a sample
    drawn = draw(
        formula="ord ~ 0",
        model="ihg",
        m=10, theta=.2,
        n=500, seed=42)

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df,
        formula="ord ~ 0",
        model="ihg",
        gen_pars={
            "theta": drawn.pars[0],
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> IHG model <<<===== ML-estimates
    =======================================================================
    m=10  Size=500  
    -----------------------------------------------------------------------
    Theta
           Estimates  StdErr    Wald  p-value
    theta      0.200  0.0086  23.292   0.0000
    =======================================================================
    Dissimilarity = 0.0639
    Loglik(sat)   = -1044.100
    Loglik(MOD)   = -1050.513
    Loglik(uni)   = -1151.293
    Mean-loglik   = -2.101
    Deviance      = 12.824
    -----------------------------------------------------------------------
    AIC = 2103.03
    BIC = 2107.24
    =======================================================================
    Elapsed time=0.00464 seconds =====>>> Fri Aug 16 12:47:55 2024
    =======================================================================

.. image:: /img/ihg0mle.png
    :alt: IHG 0 MLE

.. _ihg-with-covariates:

With covariates
^^^^^^^^^^^^^^^

.. math::
    \left\{
    \begin{array}{l}
        \Pr(R_i=1|\pmb\theta;\pmb v_i) = \theta_i
        \\
        \Pr(R_i=r+1|\pmb\theta;\pmb v_i) = \Pr(R_i=r)(1-\theta_i)\dfrac{m-r}{m-1-r(1-\theta_i)},\; r= 1, \ldots, m-1
    \end{array}
    \right.

.. math::
    \theta_i = \dfrac{1}{1 + \exp\{ - \pmb \nu_i \pmb v \}}

In this example we'll draw a sample from an IHG with covariates
and then will estimate the parameters given the observed sample.

.. code-block:: python
    :caption: Script
    :linenos:

    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods.gem import draw, estimate
    from cubmods.general import logit

    # Draw a random sample
    n = 1000
    np.random.seed(1)
    V1 = np.random.random(n)
    np.random.seed(42)
    V2 = np.random.random(n)
    df = pd.DataFrame({
        "V1": V1, "V2": V2
    })

    # draw a sample
    drawn = draw(
        df=df,
        formula="ord ~ V1 + V2",
        model="ihg",
        m=10,
        nu=[logit(.1), -2, 3],
        seed=42)

    # inferential method on drawn sample
    fit = estimate(
        df=drawn.df,
        formula=drawn.formula,
        model="ihg",
        gen_pars={
            "theta": drawn.pars[0],
        }
    )
    # print the summary of MLE
    print(fit.summary())
    # show the plot of MLE
    fit.plot()
    plt.show()

.. code-block:: none

    warnings.warn("No m given, max(ordinal) has been taken")
    =======================================================================
    =====>>> IHG(V) model <<<===== ML-estimates
    =======================================================================
    m=10  Size=1000  
    -----------------------------------------------------------------------
    Theta
              Estimates  StdErr     Wald  p-value
    constant     -2.368  0.0998  -23.741   0.0000
    V1           -1.973  0.1438  -13.721   0.0000
    V2            3.230  0.1451   22.261   0.0000
    =======================================================================
    Dissimilarity = 0.0455
    Loglik(MOD)   = -1958.475
    Loglik(uni)   = -2302.585
    Mean-loglik   = -1.958
    -----------------------------------------------------------------------
    AIC = 3922.95
    BIC = 3937.67
    =======================================================================
    Elapsed time=1.10664 seconds =====>>> Fri Aug 16 12:53:12 2024
    =======================================================================

.. image:: /img/ihgvmle.png
    :alt: IHG V MLE

MULTICUB
--------

    .. bibliography:: cub.bib
        :list: enumerated
        :filter: False

        piccolo2019class

With the **multicub** tool, parameters estimated from
multiple observed samples can be shown in a single plot.

In this example, we'll draw three samples from CUBE
models and *manually* add a shelter category. Then we'll
use the **multicub** tool for CUB models, CUBE models and
CUBSH models.

.. code-block:: python
    :caption: Script
    :linenos:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cubmods.gem import draw
    from cubmods.multicub import multi

    # draw random samples
    df = pd.DataFrame()
    for i, (pi, xi, phi) in enumerate(
        zip([.9, .8, .7], [.3, .5, .7], [.05, .1, .15])
        ):
        drawn = draw(
            formula="ord ~ 0 | 0 | 0",
            m = 9, model="cube", n=500,
            pi=pi, xi=xi, phi=phi
        )
        # add a shelter category at c=1
        df[f"ord{i+1}"] = np.concatenate((
            drawn.rv, np.repeat(1, 25)
        ))

    # MULTI-CUB
    multi(
        ords=df, ms=9, model="cub"
    )
    plt.show()
    # MULTI-CUBE
    multi(
        ords=df, ms=9, model="cube"
    )
    plt.show()
    # MULTI-CUBSH
    multi(
        ords=df, ms=9, model="cub", shs=1
    )
    plt.show()

.. image:: /img/multicub.png
    :alt: MULTICUB

.. image:: /img/multicube.png
    :alt: MULTICUBE

.. image:: /img/multicubsh.png
    :alt: MULTICUBSH