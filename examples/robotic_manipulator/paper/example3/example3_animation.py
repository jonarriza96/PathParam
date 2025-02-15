# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import make_axes

from examples.robotic_manipulator.src.utils import (
    plot_two_link_manipulator,
    load_pickle,
    get_package_path,
)

from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


def visualize(p1, p2, v_norm, eta_t_eval, v, x, u, t_sim):
    # --------------------------- Visualize trajectory --------------------------- #
    # fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # fig = plt.gcf()
    fig = plt.clf()
    plt.subplot(121)
    ax = plt.gca()
    ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.5)
    # if case == 2:
    ax.plot(bnd1[:, 0], bnd1[:, 1], "k-", alpha=0.5)
    ax.plot(bnd2[:, 0], bnd2[:, 1], "k-", alpha=0.5)
    # ax.plot(p20[0], p20[1], "go", markersize=8)
    # ax.plot(p2f[0], p2f[1], "ro", markersize=8)

    ax = plot_two_link_manipulator(ax, joint1=p1[-1], joint2=p2[-1], color="gray")

    sc = ax.scatter(
        p2[:, 0],
        p2[:, 1],
        marker=".",
        alpha=1.0,
        lw=0.5,
        cmap="turbo",
        c=v_norm,
        vmin=np.min(v_norm),
        vmax=1.2,  # np.max(v_norm) * 1.05,
    )
    # cbar = plt.colorbar(sc)
    # cbar.set_label(r"$\|v\|$", fontsize=fs)

    # Create a custom position for the colorbar
    # cbar_ax = fig.add_axes([0.3, 0.78, 0.1, 0.02])
    # cbar = plt.colorbar(
    #     sc, cax=cbar_ax, orientation="horizontal"
    # )  # Horizontal colorbar
    # cbar.set_label(r"$\|v\|$", fontsize=fs, labelpad=10)
    # cbar.set_ticks([0, 0.6, 1.2])
    # cbar.ax.xaxis.set_label_position("top")  # Move label to the top
    # cbar.ax.xaxis.set_ticks_position("bottom")  # Keep ticks at the bottom

    # Other labels
    # ax.set_xlim([-0.5, 1.3])
    ax.set_ylim([-0.53, 1.6])
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_axis_off()

    ax.set_title(f"t={t_sim[-1]:.2f} s")

    # ------------------------------ Visualize data ------------------------------ #
    # fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(4, 5))
    # ax1, ax2, ax3, ax4 = ax
    plt.subplot(433)
    ax1 = plt.gca()
    plt.subplot(436)
    ax2 = plt.gca()
    plt.subplot(439)
    ax3 = plt.gca()
    plt.subplot(4, 3, 12)
    ax4 = plt.gca()
    axs = [ax1, ax2, ax3, ax4]
    for AX in axs:
        AX.grid(axis="x", alpha=0.35)
        AX.tick_params(axis="both", which="major", labelsize=fs)

    ax1.plot(t_sim, x[:, 1])
    ax1.plot(t_sim, eta_t_eval, "k--")
    ax1.plot(t_sim, -eta_t_eval, "k--")
    ax1.set_ylabel(r"$\eta$", fontsize=fs)
    ax1.set_xticklabels([])

    ax2.plot(t_sim, v[:, 0], label=r"$v_x$")
    ax2.plot(t_sim, v[:, 1], label=r"$v_y$")
    ax2.plot([t_sim[0], t_sim[-1]], [-1, -1], "k--")
    ax2.plot([t_sim[0], t_sim[-1]], [1, 1], "k--")
    ax2.legend(fontsize=fs - 2, loc="lower left")
    ax2.set_ylabel(r"$v$", fontsize=fs)
    ax2.set_xticklabels([])

    ax3.plot(t_sim, x[:, 6], label=r"$\dot{\theta}_{1}$")
    ax3.plot(t_sim, x[:, 7], label=r"$\dot{\theta}_{2}$")
    ax3.plot([t_sim[0], t_sim[-1]], [-1, -1], "k--")
    ax3.plot([t_sim[0], t_sim[-1]], [1, 1], "k--")
    ax3.legend(fontsize=fs - 2, loc="lower left")
    ax3.set_ylabel(r"$\dot{\theta}$", fontsize=fs)
    ax3.set_xticklabels([])

    ax4.plot([t_sim[0], t_sim[-1]], [-5, -5], "k--")
    ax4.plot([t_sim[0], t_sim[-1]], [5, 5], "k--")
    ax4.plot(t_sim[:], u[:, 0], label=r"$\ddot{\theta}_{1}$")
    ax4.plot(t_sim[:], u[:, 1], label=r"$\ddot{\theta}_{2}$")
    ax4.legend(fontsize=fs - 2, loc="lower left")
    ax4.set_ylabel(r"$\ddot{\theta}$", fontsize=fs)
    ax4.set_xlabel("Time [s]", fontsize=fs)

    plt.subplots_adjust(wspace=0.0)

    # plt.tight_layout()


# Load data
data = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example3/data/example_3.pickle"
)
t_sim = data["t_sim"]
x = data["x"]
u = data["u"]
p = data["p"]
v = data["v"]
gamma_eval = data["path"]
bounds = data["bounds"]

p1 = p[0]
p2 = p[1]
eta_t_eval = bounds[0]
bnd1 = bounds[1]
bnd2 = bounds[2]


# Necessary variables
fs = 10
N = x.shape[0]
v_norm = np.linalg.norm(v[:-1, :], axis=1)

# Interpolation: Generate dense points
num_points = 2  # Number of points per segment
t = np.arange(len(p2[:-1]))  # Original parameter
interp_t = np.linspace(
    0, len(p2[:-1]) - 1, len(p2[:-1]) * num_points
)  # Interpolated parameter

# Interpolating p2 and v_norm
interp_p1 = np.column_stack(
    [interp1d(t, p1[:-1, i], kind="linear")(interp_t) for i in range(p1.shape[1])]
)
interp_p2 = np.column_stack(
    [interp1d(t, p2[:-1, i], kind="linear")(interp_t) for i in range(p2.shape[1])]
)
interp_v_norm = interp1d(t, v_norm, kind="linear")(interp_t)
interp_x = np.column_stack(
    [interp1d(t, x[:-1, i], kind="linear")(interp_t) for i in range(x.shape[1])]
)
interp_u = np.column_stack(
    [interp1d(t, u[:, i], kind="linear")(interp_t) for i in range(u.shape[1])]
)
interp_v = np.column_stack(
    [interp1d(t, v[:-1, i], kind="linear")(interp_t) for i in range(v.shape[1])]
)
interp_eta = interp1d(t, eta_t_eval[:-1], kind="linear")(interp_t)
interp_t_sim = interp1d(t, t_sim[:-1], kind="linear")(interp_t)

# fig = plt.figure()
# visualize(p1=interp_p1, p2=interp_p2, v_norm=interp_v_norm, t=t_sim[-1])


# visualize
print("Saving animations ...")
fig = plt.figure()
animation = FuncAnimation(
    fig,
    lambda i: visualize(
        p1=interp_p1[:i],
        p2=interp_p2[:i],
        v_norm=interp_v_norm[:i],
        eta_t_eval=interp_eta[:i],
        x=interp_x[:i],
        u=interp_u[:i],
        v=interp_v[:i],
        t_sim=interp_t_sim[:i],
    ),
    frames=range(1, len(interp_p1) - 1),
)
animation.save(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example3/images/"
    + "deviations"
    + ".gif",
    writer="ffmpeg",
    fps=len(interp_p1) // 4,
)
print("Done")
