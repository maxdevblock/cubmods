import sys
sys.path.append("..")

# import libraries
import matplotlib.pyplot as plt
from cubmods.gem import draw, estimate

# draw a sample
drawn = draw(
    formula="ord ~ 0 | 0 | 0",
    m=7, sh=1,
    pi=.8, xi=.4, delta=.15,
    n=1500, seed=42)

print(drawn.as_dataframe())

# inferential method on drawn sample
fit = estimate(
    df=drawn.df, sh=drawn.sh,
    formula=drawn.formula,
    ass_pars={
        "pi1": drawn.pars[0],
        "pi2": drawn.pars[1],
        "xi": drawn.pars[2],
    }
)
# print the summary of MLE
print(fit.summary())
# show the plot of MLE
fit.plot()
plt.show()