import sys
sys.path.append("..")

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
    formula="fee ~ 0 | 0 | 0",
    model="cube",
    df=df,
    m=9,
    pi=.8,
    xi=.3,
    phi=.12,
    n=df.index.size
)

# MLE estimation
fit = estimate(
    formula="fee ~ 0 | 1 | 0",
    model="cube",
    df=drawn.df,
    ass_pars={
        "pi": drawn.pars[0],
        "gamma": logit(drawn.pars[1:-1]),
        "phi": drawn.pars[-1]
    }
)
# Print MLE summary
print(fit.summary())
# plot the results
fit.plot()
plt.show()