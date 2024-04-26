Classes and functions of `smry` module.

```Python
from cubmods import smry
```

***

# Classes

## `CUBsample`

An instance of this Class is returned by `.draw()` functions. See the corresponding model's function for details.

- Methods
  - `.model` (_string_): model from which the sample has been drawn
  - `.diss` (_int_): dissimilarity index between drawn sample and generating model
  - `.theoric` (_array_): the array of the PMF (for models without covariates) or the Average Probabilty Mass (for models with covariates); it is of length `m`
  - `.m` (_int_): number of ordinal categories
  - `.sh` (_int_): if provided _shelter choice_ (or a list of 2 _choices_ if 2-cush) of the model; default is `None`
  - `.pars` (_array_): array of parameter values of the specified model
  - `.par_names` (_array_): array of parameter names of the specified model 
  - `.V` (_DataFrame_): dataframe of covariates for $\theta$ parameter of IHG model
  - `.W` (_DataFrame_): dataframe of covariates for Feeling
  - `.X` (_DataFrame_): dataframe of covariates for shelter choice; `cush2_xx` returns a list of `[X1, X2]` covariates for the 2 shelter choices
  - `.Y` (_DataFrame_): dataframe of covariates for Uncertainty
  - `.Z` (_DataFrame_): dataframe of covariates for Overdispersion 
  - `.p` (_array_): array of parameters size
  - `.rv` (_array_): drawn sample of ordinal responses 
  - `.n` (_int_): size of the drawn sample 
  - `.seed` (_int_): random seed of the drawn sample; it is `None` if no seed has been defined in `.draw()` function 
- Functions
  - `.summary()`
    
    Prints a summary of the drawn sample.
    - Arguments
      - this function takes no arguments
    - Returns
      -  A summary of the drawn sample as a string table
  - `.plot()`
    
    Plots the relative frequencies of the drawn sample and PMF (or Average Probability for models with covariates) of the generating model
    - Arguments
        - `figsize=(7, 15)` (_tuple_) a tuple of integers of figure size `(weight, height)`; see [matplotlib](https://matplotlib.org) documentation for details
        - `ci=.95` (_float_): confidence level of ellipse; must be $(0,1)$
        - `saveas=None` (_string_): filename of the plot to be saved; if `None` the plot won't be saved; must end with a supported extension (for example `fname.png`); see [matplotlib](https://matplotlib.org) documentation for details
    - Returns
        - a tuple of _(fig, ax)_; see [matplotlib](https://matplotlib.org) documentation for details
  - `.as_dataframe()`
    Returns the drawn sample as a `pandas` DataFrame
    - Arguments
      - `varname="ordinal"` (_string_): the name to be assigned at the ordinal variable in the DataFrame
    - Returns
      - A DataFrame of $n$ rows and 1 column named `varname` of the drawn sample
  - `.save(fname)`
    Save the object.
    - Arguments 
      - `fname` (_string_): name to assign at the saved object; the object is save in same directory of the script; the complete name will be `<fname>.cub.sample`
    - Returns
      - This function has no return

## `CUBres`

Default Class for MLE results; each model module function extends this Class with specific functions. An instance  of the extended Class is returned by `.mle()` functions.

- Methods
  - `.` (_type_): 

- Functions
  - `.()`

***

# Functions

## `.as_txt()`

What. #TODO: move inside `CUBres`

- Arguments
  - `arg` (_type_): 
- Returns
  - What.