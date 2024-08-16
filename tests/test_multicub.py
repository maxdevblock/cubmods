import sys
sys.path.append("..")

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