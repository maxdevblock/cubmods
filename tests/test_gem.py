import sys
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
sys.path.append("..")
from cubmods.gem import estimate, draw
from cubmods.general import logit

n = 500
m = 9
A = sps.binom(n=10, p=.3).rvs(n,
    random_state=42)
B = sps.binom(n=2, p=.6).rvs(n,
    random_state=42)
C = sps.gamma(a=2, scale=1/.5).rvs(n,
    random_state=42)
D = sps.poisson(50).rvs(n,
    random_state=42)
df = pd.DataFrame({
    "A":A, "B":B, "C":C, "D":D
})
df["B"].replace({
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e'
    }, inplace=True)

models = [
    #cub
    {"draw": dict(
        model="cub", m=m,
        formula="ord ~ 0 | 0",
        pi=.7, xi=.3,
        n=n,
        seed=76
        ),
    "est": dict(
        model="cub", m=m,
        formula="ord ~ 0 | 0",
        )
    },
    {"draw": dict(
        model="cub", m=m,
        formula="ord ~ A | 0",
        beta=[logit(.7), -2], xi=.3,
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cub", m=m,
        formula="ord ~ A | 0",
        )
    },
    {"draw": dict(
        model="cub", m=m,
        formula="ord ~ 0 | C(B)",
        pi=.7, gamma=[logit(.3),-2,-1],
        n=2000, df=df,
        seed=76
        ),
    "est": dict(
        model="cub", m=m,
        formula="ord ~ 0 | C(B)",
        )
    },
    {"draw": dict(
        model="cub", m=m,
        formula="ord ~ A+C | C(B)",
        beta=[logit(.7),-2,2],
        gamma=[logit(.3),-2,-1],
        n=2000, df=df,
        seed=76
        ),
    "est": dict(
        model="cub", m=m,
        formula="ord ~ A+C | C(B)",
        )
    },
    #cubsh
    {"draw": dict(
        model="cub", m=m, sh=1,
        formula="ord ~ 0 | 0 | 0",
        pi=.7, xi=.3, delta=.15,
        n=n,
        seed=76
        ),
    "est": dict(
        model="cub", m=m, sh=1,
        formula="ord ~ 0 | 0 | 0",
        )
    },
    {"draw": dict(
        model="cub", m=m, sh=1,
        formula="ord ~ D | A | C(B)",
        beta=[logit(.7),2],
        gamma=[logit(.3),2],
        omega=[logit(.15),2,-2],
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cub", m=m, sh=1,
        formula="ord ~ D | A | C(B)",
        )
    },
    #cush
    {"draw": dict(
        model="cush", m=m, sh=1,
        formula="ord ~ 0",
        delta=.15,
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cush", m=m, sh=1,
        formula="ord ~ 0",
        )
    },
    {"draw": dict(
        model="cush", m=m, sh=1,
        formula="ord ~ A+C(B)",
        omega=[logit(.15),-2,2,-3],
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cush", m=m, sh=1,
        formula="ord ~ A+C(B)",
        )
    },
    #cush2
    {"draw": dict(
        model="cush2", m=m, sh=[1,3],
        formula="ord ~ 0 | 0",
        delta1=.15,
        delta2=.2,
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cush2", m=m, sh=[1,3],
        formula="ord ~ 0 | 0",
        )
    },
    {"draw": dict(
        model="cush2", m=m, sh=[1,3],
        formula="ord ~ D+C | 0",
        omega1=[logit(.15),.02,-.1],
        delta2=.2,
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cush2", m=m, sh=[1,3],
        formula="ord ~ D+C | 0",
        )
    },
    #ihg
    {"draw": dict(
        model="ihg", m=m,
        formula="ord ~ 0",
        theta=.2,
        n=n,
        seed=76
        ),
    "est": dict(
        model="ihg", m=m,
        formula="ord ~ 0",
        )
    },
    {"draw": dict(
        model="ihg", m=m,
        formula="ord ~ C(B)",
        nu=[logit(.2),-1,1],
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="ihg", m=m,
        formula="ord ~ C(B)",
        )
    },
    #cube
    {"draw": dict(
        model="cube", m=m,
        formula="ord ~ 0 | 0 | 0",
        pi=.7, xi=.3, phi=.1,
        n=n,
        seed=76
        ),
    "est": dict(
        model="cube", m=m,
        formula="ord ~ 0 | 0 | 0",
        )
    },
    #cube
    {"draw": dict(
        model="cube", m=m,
        formula="ord ~ 0 | D | 0",
        pi=.7, gamma=[logit(.3),3], phi=.1,
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cube", m=m,
        formula="ord ~ 0 | D | 0",
        )
    },
    #cube
    {"draw": dict(
        model="cube", m=m,
        formula="ord ~ A | D | C",
        beta=[logit(.7),2],
        gamma=[logit(.3),3],
        alpha=[np.exp(-.1),2],
        n=n, df=df,
        seed=76
        ),
    "est": dict(
        model="cube", m=m,
        formula="ord ~ A | D | C",
        )
    },
]

for mod in models:
    modname = mod["draw"]["model"]
    modform = mod["draw"]["formula"]
    print("***************************************")
    print(f"******* {modname} {modform}")
    print("***************************************\n")
    print("  draw")
    drawn = draw(**mod["draw"])
    print(drawn.summary())
    drawn.plot()
    plt.show()
    print(f"    ok {drawn.rv.size}")
    #print("    "+drawn.df.columns)
    print("  estimate")
    estim = estimate(**mod["est"],
        df=drawn.df)
    print(estim.summary())
    estim.plot()
    plt.show()
    print("    ok")
