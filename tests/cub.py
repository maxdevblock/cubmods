import sys
sys.path.append("..")

# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(
    formula="ord ~ 0 | 0",
    m=10, pi=.7, xi=.2,
    n=500, seed=1)
# print the summary of the drawn sample
print(drawn.summary())
# show the plot of the drawn sample
drawn.plot()
plt.show()

print(drawn.as_dataframe())

# inferential method on drawn sample
fit = estimate(
    df=drawn.df,
    formula="ord~0|0",
    ass_pars={
        "pi": drawn.pars[0],
        "xi": drawn.pars[1]
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()