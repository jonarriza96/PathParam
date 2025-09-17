# %%
import numpy as np
import matplotlib.pyplot as plt

from pathparam.visualize import plot_frames


from examples.robotic_manipulator.src.utils import load_pickle, get_package_path


data = load_pickle(
    "/Users/jonarrizabalaga/PathParam/examples/dubins/paper/data/dubins.pickle"
)

x = data["x"]
u = data["u"]
xi = data["xi"]
eta = data["eta"]
theta = data["theta"]
theta_dot = data["theta_dot"]
xidot = data["xidot"]
p = data["p"]

start = data["start"]
goal = data["goal"]
gamma_wp = data["gamma_wp"]
rho = data["rho"]
eta_max = data["eta_max"]
path_eval = data["path_eval"]
p_dubins = data["p_dubins"]


# %%
fs = 14

# -------------------------------- Trajectory -------------------------------- #
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

arrow_scale = 0.4
plt.arrow(
    start[0],
    start[1],
    arrow_scale * np.cos(start[2]),
    arrow_scale * np.sin(start[2]),
    head_width=0.15,
    head_length=0.15,
    fc="green",
    ec="green",
    linewidth=2,
)
plt.arrow(
    goal[0],
    goal[1],
    arrow_scale * np.cos(goal[2]),
    arrow_scale * np.sin(goal[2]),
    head_width=0.15,
    head_length=0.15,
    fc="red",
    ec="red",
    linewidth=2,
)

circle1 = plt.Circle(
    (start[0] - rho * np.sin(start[2]), start[1] + rho * np.cos(start[2])),
    rho,
    fill=False,
    color="green",
    alpha=0.3,
    linestyle="--",
    # label="Start circle",
)
circle2 = plt.Circle(
    (goal[0] - rho * np.sin(goal[2]), goal[1] + rho * np.cos(goal[2])),
    rho,
    fill=False,
    color="red",
    alpha=0.3,
    linestyle="--",
    # label="Goal circle",
)
ax.add_patch(circle1)
ax.add_patch(circle2)

ax.scatter(
    [start[0]], [start[1]], c="green", s=200, label="Start", marker="o", zorder=10
)
ax.scatter([goal[0]], [goal[1]], c="red", s=200, label="Goal", marker="o", zorder=10)
# ax.plot(gamma_wp[:, 0], gamma_wp[:, 1], color='black', alpha=0.25, linestyle='--', marker='o', label='Linear γ')
# ax.plot(gamma[:, 0], gamma[:, 1], color='m', alpha=0.5, linestyle='-', label='Smooth γ')
plot_frames(
    ax=ax,
    r=path_eval["gamma"],
    e1=path_eval["e1"],
    e2=path_eval["e2"],
    e3=np.zeros_like(path_eval["e1"]),
    scale=0.1,
    interval=0.98,
    ax_equal=False,
    planar=True,
)
ax.plot(p[:, 0], p[:, 1], "b-", linewidth=2, label=f"Dubins Path", zorder=5)
# ax.plot(gamma_wp[:, 0], gamma_wp[:, 1], "k--", alpha=0.5)

ax.set_aspect("equal")
ax.axis("off")
# ax.legend(columnspacing=1.5, ncols=5, loc="upper left")
ax.legend(labelspacing=0.8)

plt.savefig(
    get_package_path() + "/examples/dubins/paper/images/dubins_trajectory.pdf", dpi=300
)

# %%
# ----------------------------------- Data ----------------------------------- #
# fig = plt.figure()
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(4, 5))
ax1, ax2, ax3 = ax
for AX in ax:
    AX.grid(axis="x", alpha=0.35)
    AX.tick_params(axis="both", which="major", labelsize=fs)


ax1.plot(xi, eta, "b-", linewidth=2)
ax1.set_ylabel(r"$\eta$", fontsize=fs)

ax2.plot(xi, xidot, "b-", linewidth=2)
ax2.set_ylabel(r"$\dot{\xi}$", fontsize=fs)

ax3.plot(xi[:-1], theta_dot, "b-", linewidth=2)  # ,linewidth=3)
ax3.plot([xi[0], xi[-1]], [-1, -1], "k--", alpha=0.75)
ax3.plot([xi[0], xi[-1]], [1, 1], "k--", alpha=0.75)
ax3.set_ylabel(r"$\dot{\theta}$", fontsize=fs)
ax3.set_xlabel(r"$\xi$", fontsize=fs)

plt.tight_layout()

plt.savefig(
    get_package_path() + "/examples/dubins/paper/images/dubins_data.pdf", dpi=300
)
