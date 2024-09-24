import sys
sys.path.append("..")

# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(
    formula="ord ~ 0 | 0 | 0",
    model="cube",
    m=9, pi=.7, xi=.3, phi=.15,
    n=500, seed=1)

# inferential method on drawn sample
fit = estimate(
    df=drawn.df,
    formula="ord~0|0|0",
    model="cube",
    ass_pars={
        "pi": drawn.pars[0],
        "xi": drawn.pars[1],
        "phi": drawn.pars[2],
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()