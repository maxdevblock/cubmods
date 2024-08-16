import sys
sys.path.append("..")

# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(
    formula="ord ~ 0 | 0",
    model="cush",
    sh=[1,7],
    m=7,
    delta1=.15, delta2=.1,
    n=1000, seed=42)

# inferential method on drawn sample
fit = estimate(
    df=drawn.df,
    model="cush",
    formula="ord~0|0",
    sh=[1,7],
    gen_pars={
        "delta1": drawn.pars[0],
        "delta2": drawn.pars[1],
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()