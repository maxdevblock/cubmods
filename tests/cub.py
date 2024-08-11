import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from cubmods import gem, cub_0w

# Draw a random sample
n = 1000
np.random.seed(1)
W1 = np.random.randint(1, 10, n)
np.random.seed(42)
W2 = np.random.randint(1, 10, n)
df = pd.DataFrame({
    "W1": W1, "W2": W2
})
drawn = cub_0w.draw(m=10, n=n, 
    pi=0.8,
    gamma=[2.3, -0.4, -0.05],
    W=df
)
drawn.plot()
plt.show()

# add the drawn sample
df["ordinal"] = drawn.rv
# MLE estimation
mod1 = gem.from_formula(
    formula="ordinal ~ 0 | W1+W2 | 0",
    df=df,
)
# Print MLE summary
print(mod1.summary())
# plot the results
mod1.plot()
plt.show()