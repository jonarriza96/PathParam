# %%
from robotic_manipulator.utils import (
    get_package_path,
    load_pickle,
    plot_two_link_manipulator,
    get_colors,
)

import numpy as np
import matplotlib.pyplot as plt


color_g, color_p = get_colors()

# --------------------------------- Overview --------------------------------- #
# Load data
data = load_pickle(
    get_package_path() + "/paper/example1/data/following_nodisturbance.pickle"
)
X = data["X"]
U = data["U"]
t_sim = data["t_sim"]
path = data["path"]

# visualize
fig, ax = plt.subplots()
ax.plot(path[:, 0], path[:, 1], "k--", alpha=1.0, linewidth=1.2)
for k in np.linspace(0, X.shape[0] - 1, num=3, dtype=int):
    ax = plot_two_link_manipulator(ax, joint1=X[k, 0:2], joint2=X[k, 2:4], color="g")
ax.plot(X[:, 2], X[:, 3], "-", color=color_g, alpha=0.75, lw=1.5)
# ax.plot(X[:, 0], X[:, 1], "-", color="gray", alpha=1.0, lw=0.5)
ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.axis("off")

plt.savefig(get_package_path() + "/paper/example1/images/overview.pdf", dpi=300)


# -------------------------------- Comparison -------------------------------- #
# Load data
data_following = load_pickle(
    get_package_path() + "/paper/example1/data/following_disturbance.pickle"
)
X_following = data_following["X"]
U_following = data_following["U"]
t_sim_following = data_following["t_sim"]

data_tracking = load_pickle(
    get_package_path() + "/paper/example1/data/tracking_disturbance.pickle"
)
X_tracking = data_tracking["X"]
U_tracking = data_tracking["U"]
t_sim_tracking = data_tracking["t_sim"]


# visualize
fig, ax = plt.subplots()
ax.plot(path[:, 0], path[:, 1], "k--", alpha=1.0, linewidth=1.2)
ax.plot(X_tracking[:, 2], X_tracking[:, 3], "-", color=color_p, alpha=0.75, lw=1.5)
ax.plot(X_following[:, 2], X_following[:, 3], "-", color=color_g, alpha=0.75, lw=1.5)
ax.set_aspect("equal")
plt.axis("off")

plt.savefig(get_package_path() + "/paper/example1/images/overview.pdf", dpi=300)

plt.show()
