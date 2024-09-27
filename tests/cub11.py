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
W1 = np.random.randint(1, 10, n)
np.random.seed(42)
W2 = np.random.random(n)
df = pd.DataFrame({
    "W1": W1, "W2": W2
})
drawn = draw(
    formula="res ~ 1 | 1",
    df=df,
    m=10, n=n,
    beta=[logit(.8)],
    gamma=[logit(.7)],
)
# print the summary
print(drawn.summary())

# plot the drawn sample
drawn.plot()
plt.show()

# print the parameters' values
print(drawn.as_dataframe())

# print the updated DataFrame
print(drawn.df)

# MLE estimation
fit = estimate(
    formula=drawn.formula,
    df=drawn.df,
    ass_pars={
        "beta": drawn.pars[:1],
        "gamma": drawn.pars[1:]
    }
)
# Print MLE summary
print(fit.summary())
# plot the results
fit.plot()
plt.show()