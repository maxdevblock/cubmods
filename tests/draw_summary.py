import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from cubmods.gem import draw
from cubmods.general import logit

n = 200
models = [
    {
        "family": "cub",
        "formula": "ord ~ 0|0",
        "params": {
            "pi": .3,
            "xi": .7
        },
        "df": None
    },
    {
        "family": "cube",
        "formula": "ord ~ 0|0|0",
        "params": {
            "pi": .3,
            "xi": .7,
            "phi": .1,
        },
        "df": None
    },
    {
        "family": "cub",
        "formula": "ord ~ 0|0|0",
        "params": {
            "pi": .3,
            "xi": .7,
            "delta": .1,
            "sh": 2
        },
        "df": None
    },
    {
        "family": "cush",
        "formula": "ord ~ 0",
        "params": {
            "delta": .3,
            "sh": 3,
        },
        "df": None
    },
    {
        "family": "cush2",
        "formula": "ord ~ 0|0",
        "params": {
            "delta1": .3,
            "delta2": .1,
            "sh": [2, 5],
        },
        "df": None
    },
    {
        "family": "ihg",
        "formula": "ord ~ 0",
        "params": {
            "theta": .25,
        },
        "df": None
    },
    # with covariates
    {
        "family": "cub",
        "formula": "ord ~ 0|A",
        "params": {
            "pi": .3,
            "gamma": [logit(.7), logit(.4)]
        },
        "df": pd.DataFrame({
            "A": np.random.random(n)
        })
    },
    {
        "family": "cub",
        "formula": "ord ~ A|0",
        "params": {
            "beta": [logit(.3), logit(.6)],
            "xi": .3
        },
        "df": pd.DataFrame({
            "A": np.random.random(n)
        })
    },
    {
        "family": "cub",
        "formula": "ord ~ A|B",
        "params": {
            "beta": [logit(.3), logit(.6)],
            "gamma": [logit(.7), logit(.4)]
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "B": np.random.random(n)
        })
    },
    {
        "family": "cube",
        "formula": "ord ~ 0|A|0",
        "params": {
            "pi": .7,
            "gamma": [logit(.7), logit(.4)],
            "phi": .11
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "B": np.random.random(n)
        })
    },
    {
        "family": "cube",
        "formula": "ord ~ A|B|C+A",
        "params": {
            "beta": [logit(.3), logit(.6)],
            "gamma": [logit(.7), logit(.4)],
            "alpha": [np.exp(.1), np.exp(.03), np.exp(.15)]
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "B": np.random.random(n),
            "C": np.random.random(n)
        })
    },
    {
        "family": "cub",
        "formula": "ord ~ A|B|C+A",
        "params": {
            "beta": [logit(.3), logit(.6)],
            "gamma": [logit(.7), logit(.4)],
            "omega": [logit(.1), logit(.05), logit(.15)],
            "sh": 1,
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "B": np.random.random(n),
            "C": np.random.random(n)
        })
    },
    {
        "family": "cush",
        "formula": "ord ~ C+A",
        "params": {
            "omega": [logit(.1), logit(.05), logit(.15)],
            "sh": 4,
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "C": np.random.random(n)
        })
    },
    {
        "family": "cush",
        "formula": "ord ~ C+A|0",
        "params": {
            "omega1": [logit(.1), logit(.05), logit(.15)],
            "delta2": .3,
            "sh": [2, 6],
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "C": np.random.random(n)
        })
    },
    {
        "family": "cush",
        "formula": "ord ~ A|C+A",
        "params": {
            "omega1": [logit(.2), logit(.1)],
            "omega2": [logit(.1), logit(.05), logit(.15)],
            "sh": [2, 6],
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "C": np.random.random(n)
        })
    },
    {
        "family": "ihg",
        "formula": "ord ~ C+A",
        "params": {
            "nu": [logit(.1), logit(.05), logit(.15)],
        },
        "df": pd.DataFrame({
            "A": np.random.random(n),
            "C": np.random.random(n)
        })
    },
]

for model in models:
    drawn = draw(
        model=model["family"],
        formula=model["formula"],
        df=model["df"],
        **model["params"],
    )
    print(drawn.summary())