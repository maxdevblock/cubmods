import sys
sys.path.append("..")

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
    ass_pars={
        "omega1": drawn.pars[:2],
        "omega2": [logit(drawn.pars[-1])]
    }
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