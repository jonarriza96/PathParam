# %%
import numpy as np
import matplotlib.pyplot as plt

import casadi as cs
from examples.robotic_manipulator.src.utils import load_pickle, get_package_path

from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def xidot_funcs():
    # define path
    xi = cs.SX.sym("xi")
    xi_dot0 = 0.15
    xi_dot1 = 0.05
    xi_dot2 = 0.12 + 0.02 * cs.sin(3 * 2 * np.pi * xi)
    xi_dot3 = 0.08 + 0.02 * cs.cos(6 * 2 * np.pi * xi)
    f_xidot = cs.Function([xi], [xi_dot0, xi_dot1, xi_dot2, xi_dot3])
    return f_xidot


def visualize(p_c0, p_c1, p_c2, p_c3, X_c0, X_c1, X_c2, X_c3):
    fig = plt.clf()

    plt.subplot(121)
    ax = plt.gca()
    ax.plot(path[:, 0], path[:, 1], "k--", alpha=1.0, linewidth=1.2)
    ax.plot(p_c0[:, 0], p_c0[:, 1], "-", alpha=1.0, lw=1.5)
    ax.plot(p_c0[-1, 0], p_c0[-1, 1], "o", markersize=10, color=colors[0])
    ax.plot(p_c1[:, 0], p_c1[:, 1], "-", alpha=1.0, lw=1.5)
    ax.plot(p_c1[-1, 0], p_c1[-1, 1], "o", markersize=10, color=colors[1])
    ax.plot(p_c2[:, 0], p_c2[:, 1], "-", alpha=1.0, lw=1.5)
    ax.plot(p_c2[-1, 0], p_c2[-1, 1], "o", markersize=10, color=colors[2])
    ax.plot(p_c3[:, 0], p_c3[:, 1], "-", alpha=1.0, lw=1.5)
    ax.plot(p_c3[-1, 0], p_c3[-1, 1], "o", markersize=10, color=colors[3])
    ax.set_aspect("equal")
    ax.axis("off")

    plt.subplot(122)
    ax = plt.gca()
    ax.plot(xi_eval, xidot0_eval, "--", color=colors[0], alpha=0.5)
    ax.plot(X_c0[:, 0], X_c0[:, 2], color=colors[0])
    ax.plot(X_c0[-1, 0], X_c0[-1, 2], "o", markersize=10, color=colors[0])
    ax.plot(xi_eval, xidot1_eval, "--", color=colors[1], alpha=0.5)
    ax.plot(X_c1[:, 0], X_c1[:, 2], color=colors[1])
    ax.plot(X_c1[-1, 0], X_c1[-1, 2], "o", markersize=10, color=colors[1])
    ax.plot(xi_eval, xidot2_eval, "--", color=colors[2], alpha=0.5)
    ax.plot(X_c2[:, 0], X_c2[:, 2], color=colors[2])
    ax.plot(X_c2[-1, 0], X_c2[-1, 2], "o", markersize=10, color=colors[2])
    ax.plot(xi_eval, xidot3_eval, "--", color=colors[3], alpha=0.5)
    ax.plot(X_c3[:, 0], X_c3[:, 2], color=colors[3])
    ax.plot(X_c3[-1, 0], X_c3[-1, 2], "o", markersize=10, color=colors[3])
    ax.set_xlabel(r"$\xi$", fontsize=14)
    ax.set_ylabel(r"$\dot{\xi}$", fontsize=14)
    ax.set_suptitle("Velocity Profile", fontsize=14)
    # ax.tick_params(axis="both", which="major", labelsize=13)

    plt.tight_layout()


def upsample_vector(vector, n_desired):
    n = len(vector)
    x = np.arange(n)
    f = interp1d(x, vector, kind="linear")
    x_new = np.linspace(0, n - 1, n_desired)
    return f(x_new)


# Load data
data_c0 = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example2/data/case_0.pickle"
)
X_c0 = data_c0["X"]
U_c0 = data_c0["U"]
p_c0 = data_c0["p"]
t_sim_c0 = data_c0["t_sim"]
path = data_c0["path"]

data_c1 = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example2/data/case_1.pickle"
)
X_c1 = data_c1["X"]
U_c1 = data_c1["U"]
p_c1 = data_c1["p"]
t_sim_c1 = data_c1["t_sim"]
path = data_c1["path"]

data_c2 = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example2/data/case_2.pickle"
)
X_c2 = data_c2["X"]
U_c2 = data_c2["U"]
p_c2 = data_c2["p"]
t_sim_c2 = data_c2["t_sim"]

data_c3 = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example2/data/case_3.pickle"
)
X_c3 = data_c3["X"]
U_c3 = data_c3["U"]
p_c3 = data_c3["p"]
t_sim_c3 = data_c3["t_sim"]

# upsample so that all have same length
n_iters = max(len(p_c0), len(p_c1), len(p_c2), len(p_c3))
p_c0 = np.vstack(
    [upsample_vector(p_c0[:, 0], n_iters), upsample_vector(p_c0[:, 1], n_iters)]
).T
p_c1 = np.vstack(
    [upsample_vector(p_c1[:, 0], n_iters), upsample_vector(p_c1[:, 1], n_iters)]
).T
p_c2 = np.vstack(
    [upsample_vector(p_c2[:, 0], n_iters), upsample_vector(p_c2[:, 1], n_iters)]
).T
p_c3 = np.vstack(
    [upsample_vector(p_c3[:, 0], n_iters), upsample_vector(p_c3[:, 1], n_iters)]
).T

X_c0_ups = []
X_c1_ups = []
X_c2_ups = []
X_c3_ups = []
for i in range(7):
    X_c0_ups.append(upsample_vector(X_c0[:, i], n_iters))
    X_c1_ups.append(upsample_vector(X_c1[:, i], n_iters))
    X_c2_ups.append(upsample_vector(X_c2[:, i], n_iters))
    X_c3_ups.append(upsample_vector(X_c3[:, i], n_iters))
X_c0_ups = np.squeeze([X_c0])
X_c1_ups = np.squeeze([X_c1])
X_c2_ups = np.squeeze([X_c2])
X_c3_ups = np.squeeze([X_c3])


# evaluate references
# n_eval = 1000
f_xidot0 = lambda xi: 0.15 * np.ones_like(xi)
f_xidot1 = lambda xi: 0.05 * np.ones_like(xi)
f_xidot2 = lambda xi: 0.12 + 0.02 * np.sin(3 * 2 * np.pi * xi)
f_xidot3 = lambda xi: 0.08 + 0.02 * np.cos(6 * 2 * np.pi * xi)

xi_eval = np.linspace(0, 1, n_iters)
xidot0_eval = f_xidot0(xi_eval)
xidot1_eval = f_xidot1(xi_eval)
xidot2_eval = f_xidot2(xi_eval)
xidot3_eval = f_xidot3(xi_eval)

# visualize
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig = plt.figure()
# visualize(p_c0, p_c1, p_c2, p_c3, X_c0, X_c1, X_c2, X_c3)

# visualize
print("Saving animations ...")
fig = plt.figure()
animation = FuncAnimation(
    fig,
    lambda i: visualize(
        p_c0[:i, :],
        p_c1[:i, :],
        p_c2[:i, :],
        p_c3[:i, :],
        X_c0_ups[:i, :],
        X_c1_ups[:i, :],
        X_c2_ups[:i, :],
        X_c3_ups[:i, :],
    ),
    frames=range(2, len(p_c0) - 1),
)
animation.save(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example2/images/"
    + "v_profile"
    + ".gif",
    writer="ffmpeg",
    fps=n_iters // 4,
)
print("Done")

plt.show()
