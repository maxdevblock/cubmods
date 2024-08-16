import sys
sys.path.append("..")

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