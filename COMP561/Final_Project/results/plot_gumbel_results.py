import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

monte_carlo = pd.read_csv("results/monte_carlo_results.csv")
monte_carlo = monte_carlo[monte_carlo["n"] == 40]

hist = np.histogram(monte_carlo["score"].values, bins=10, density=True)

print(hist)
fig, ax = plt.subplots(1, 1)

ax.bar(hist[1][:-1], hist[0], alpha=0.6, width=0.3, linewidth=3)

loc, scale = ss.gumbel_r.fit(monte_carlo["score"].values)
raise Exception(loc, scale)
dist = ss.gumbel_r(loc=loc, scale=scale)
ax.plot(
    sorted(monte_carlo["score"].values),
    dist.pdf(sorted(monte_carlo["score"].values)),
    "r-",
)
ax.set_xlabel("Score from Smith-waterman Alignment")
ax.set_ylabel("Probability Density")
ax.set_title("Gumbel Distribution Fit for query length = 50")
fig.savefig("results/gumbel_plot.png")
