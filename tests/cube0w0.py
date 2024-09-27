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
    ass_pars={
        "pi": drawn.pars[0],
        "gamma": drawn.pars[1:-1],
        "phi": drawn.pars[-1]
    }
)
# Print MLE summary
print(fit.summary())
# plot the results
fit.plot()
plt.show()

# MLE estimation
fit = estimate(
    formula="fee ~ 1 | W | 1",
    model="cube",
    df=drawn.df,
    ass_pars={
        "beta": [logit(drawn.pars[0])],
        "gamma": drawn.pars[1:3],
        "alpha": [np.log(drawn.pars[3])]
    }
)
# Print MLE summary
print(fit.summary())
# plot the results
fit.plot()
plt.show()

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