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

