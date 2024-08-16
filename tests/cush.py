import sys
sys.path.append("..")

# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(
    formula="ord ~ 0",
    model="cush",
    sh=7,
    m=7, delta=.15,
    n=1500, seed=76)

# inferential method on drawn sample
fit = estimate(
    df=drawn.df,
    model="cush",
    formula="ord~0",
    sh=7,
    gen_pars={
        "delta": drawn.pars[0],
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()