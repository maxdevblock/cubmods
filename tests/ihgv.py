import sys
sys.path.append("..")

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
    ass_pars={
        "theta": drawn.pars[0],
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()