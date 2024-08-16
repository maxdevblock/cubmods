import sys
sys.path.append("..")

# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(
    formula="ord ~ 0",
    model="ihg",
    m=10, theta=.2,
    n=500, seed=42)

# inferential method on drawn sample
fit = estimate(
    df=drawn.df,
    formula="ord ~ 0",
    model="ihg",
    gen_pars={
        "theta": drawn.pars[0],
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()
