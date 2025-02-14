import numpy as np
import matplotlib.pyplot as plt

import casadi as cs
from robotic_manipulator.utils import load_pickle, get_package_path


def xidot_funcs():
    # define path
    xi = cs.SX.sym("xi")
    xi_dot0 = 0.15
    xi_dot1 = 0.05
    xi_dot2 = 0.12 + 0.02 * cs.sin(3 * 2 * np.pi * xi)
    xi_dot3 = 0.08 + 0.02 * cs.cos(6 * 2 * np.pi * xi)
    f_xidot = cs.Function([xi], [xi_dot0, xi_dot1, xi_dot2, xi_dot3])
    return f_xidot


# Load data
data_c0 = load_pickle(get_package_path() + "/paper/example2/data/case_0.pickle")
X_c0 = data_c0["X"]
X_ref_c0 = data_c0["X_ref"]
U_c0 = data_c0["U"]
p_c0 = data_c0["p"]
t_sim_c0 = data_c0["t_sim"]
path = data_c0["path"]

data_c1 = load_pickle(get_package_path() + "/paper/example2/data/case_1.pickle")
X_c1 = data_c1["X"]
X_ref_c1 = data_c1["X_ref"]
U_c1 = data_c1["U"]
p_c1 = data_c1["p"]
t_sim_c1 = data_c1["t_sim"]
path = data_c1["path"]

data_c2 = load_pickle(get_package_path() + "/paper/example2/data/case_2.pickle")
X_c2 = data_c2["X"]
X_ref_c2 = data_c2["X_ref"]
U_c2 = data_c2["U"]
p_c2 = data_c2["p"]
t_sim_c2 = data_c2["t_sim"]

data_c3 = load_pickle(get_package_path() + "/paper/example2/data/case_3.pickle")
X_c3 = data_c3["X"]
X_ref_c3 = data_c3["X_ref"]
U_c3 = data_c3["U"]
p_c3 = data_c3["p"]
t_sim_c3 = data_c3["t_sim"]

n_eval = 1000
f_xidot0 = lambda xi: 0.15 * np.ones_like(xi)
f_xidot1 = lambda xi: 0.05 * np.ones_like(xi)
f_xidot2 = lambda xi: 0.12 + 0.02 * np.sin(3 * 2 * np.pi * xi)
f_xidot3 = lambda xi: 0.08 + 0.02 * np.cos(6 * 2 * np.pi * xi)

xi_eval = np.linspace(0, 1, n_eval)
xidot0_eval = f_xidot0(xi_eval)
xidot1_eval = f_xidot1(xi_eval)
xidot2_eval = f_xidot2(xi_eval)
xidot3_eval = f_xidot3(xi_eval)

# visualize
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots()
ax.plot(path[:, 0], path[:, 1], "k--", alpha=1.0, linewidth=1.2)
ax.plot(p_c0[:, 0], p_c0[:, 1], "-", alpha=1.0, lw=1.5)
ax.plot(p_c0[0, 0], p_c0[0, 1], "o", markersize=10, color=colors[0])
ax.plot(p_c1[:, 0], p_c1[:, 1], "-", alpha=1.0, lw=1.5)
ax.plot(p_c1[0, 0], p_c1[0, 1], "o", markersize=10, color=colors[1])
ax.plot(p_c2[:, 0], p_c2[:, 1], "-", alpha=1.0, lw=1.5)
ax.plot(p_c2[0, 0], p_c2[0, 1], "o", markersize=10, color=colors[2])
ax.plot(p_c3[:, 0], p_c3[:, 1], "-", alpha=1.0, lw=1.5)
ax.plot(p_c3[0, 0], p_c3[0, 1], "o", markersize=10, color=colors[3])
ax.set_aspect("equal")
ax.axis("off")
plt.savefig(get_package_path() + "/paper/example2/images/overview.pdf", dpi=300)


fig, ax = plt.subplots()
ax.plot(xi_eval, xidot0_eval, "--", color=colors[0], alpha=0.5)
ax.plot(X_ref_c0[:, 0], X_c0[:-1, 2], color=colors[0])
ax.plot(X_ref_c0[0, 0], X_c0[0, 2], "o", markersize=10, color=colors[0])
ax.plot(xi_eval, xidot1_eval, "--", color=colors[1], alpha=0.5)
ax.plot(X_ref_c1[:, 0], X_c1[:-1, 2], color=colors[1])
ax.plot(X_ref_c1[0, 0], X_c1[0, 2], "o", markersize=10, color=colors[1])
ax.plot(xi_eval, xidot2_eval, "--", color=colors[2], alpha=0.5)
ax.plot(X_ref_c2[:, 0], X_c2[:-1, 2], color=colors[2])
ax.plot(X_ref_c2[0, 0], X_c2[0, 2], "o", markersize=10, color=colors[2])
ax.plot(xi_eval, xidot3_eval, "--", color=colors[3], alpha=0.5)
ax.plot(X_ref_c3[:, 0], X_c3[:-1, 2], color=colors[3])
ax.plot(X_ref_c3[0, 0], X_c3[0, 2], "o", markersize=10, color=colors[3])
ax.set_xlabel(r"$\xi$", fontsize=14)
ax.set_ylabel(r"$\dot{\xi}$", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=13)
plt.savefig(get_package_path() + "/paper/example2/images/xidot.pdf", dpi=300)

plt.tight_layout()
plt.show()
