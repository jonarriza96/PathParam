# %%
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs

import pickle
import argparse
from sys import exit

from examples.robotic_manipulator.src.model import (
    forward_kinematics,
    inverse_kinematics,
    forward_kinematics,
    robotic_manipulator_spatial,
    robotic_manipulator_ffullspatial,
    spatial_to_cartesian,
    calculate_xidot,
    RK4,
)
from examples.robotic_manipulator.src.utils import (
    plot_two_link_manipulator,
    save_pickle,
)

from scipy.interpolate import CubicSpline


def generate_path_functions():
    # define path
    xi = cs.SX.sym("xi")
    gamma = cs.vertcat(xi, 0.25 * cs.sin(2 * np.pi * xi) + 1)

    gamma_d = cs.jacobian(gamma, xi)
    sigma = cs.norm_2(gamma_d)
    e1 = gamma_d / sigma
    e1_d = cs.jacobian(e1, xi)
    # e2 = e1_d / cs.norm_2(e1_d)
    e2 = cs.vertcat(-e1[1], e1[0])
    omega = cs.dot(e1_d, e2)

    sigma_d = cs.jacobian(sigma, xi)
    omega_d = cs.jacobian(omega, xi)
    e1_d = cs.jacobian(e1, xi)
    e2_d = cs.jacobian(e2, xi)

    f_gamma = cs.Function("f_gamma", [xi], [gamma])
    f_e1 = cs.Function("f_e1", [xi], [e1, e1_d], ["xi"], ["e1", "e1_d"])
    f_e2 = cs.Function("f_e2", [xi], [e2, e2_d], ["xi"], ["e2", "e2_d"])
    f_sigma = cs.Function(
        "f_sigma", [xi], [sigma, sigma_d], ["xi"], ["sigma", "sigma_d"]
    )
    f_omega = cs.Function(
        "f_omega", [xi], [omega, omega_d], ["xi"], ["omega", "omega_d"]
    )
    # if case == 0 or case == 1:
    #     f_xidot = cs.Function(
    #         "f_xidot", [xi], [0.08 + 0.02 * cs.sin(2 * 2 * np.pi * xi)]
    #     )
    # elif case == 2:
    #     f_xidot = cs.Function("f_xidot", [xi], [0.12])
    path = {
        "gamma": f_gamma,
        "e1": f_e1,
        "e2": f_e2,
        "sigma": f_sigma,
        "omega": f_omega,
        # "xi_dot": f_xidot,
        # "eta_min": -0.05,
        # "eta_max": 0.05,
    }

    # evaluate path
    n_eval = 1000
    xi_eval = np.linspace(0, 1, n_eval)
    gamma_eval = np.zeros((n_eval, 2))
    xidot_eval = np.zeros(n_eval)
    xi_ref = np.zeros(n_eval)
    eta_ref = np.zeros(n_eval)
    xidot_ref = np.zeros(n_eval)
    theta1_ref = np.zeros(n_eval)
    theta2_ref = np.zeros(n_eval)
    for k in range(n_eval):
        gamma_eval[k, :] = np.squeeze(f_gamma(xi_eval[k]))

        p2_ref_k = np.squeeze(f_gamma(xi_eval[k]))
        p1_ref_k, theta1_ref_k, theta2_ref_k = inverse_kinematics(p2_ref_k)

        xi_ref[k] = xi_eval[k]  # p2_ref_k[0]
        eta_ref[k] = 0  # p2_ref_k[1]
        # xidot_ref[k] = np.squeeze(f_xidot(xi_eval[k]))
        theta1_ref[k] = theta1_ref_k
        theta2_ref[k] = theta2_ref_k

    # xidot_spl = CubicSpline(xi_eval, xidot_ref)
    xi_spl = CubicSpline(xi_eval, xi_ref)
    eta_spl = CubicSpline(xi_eval, eta_ref)
    theta1_spl = CubicSpline(xi_eval, theta1_ref)
    theta2_spl = CubicSpline(xi_eval, theta2_ref)

    f_ref = lambda xi: np.array(
        [
            xi_spl(xi),
            eta_spl(xi),
            # xidot_spl(xi),
            0,
            theta1_spl(xi),
            theta2_spl(xi),
            theta1_spl(xi, 1),
            theta2_spl(xi, 1),
            theta1_spl(xi, 2),
            theta2_spl(xi, 2),
        ]
    ).T

    return f_ref, path


def get_bound():
    out_d = 0.05
    in_d = 0.025
    xi_knots = np.array(
        [
            0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1,
        ]
    )
    eta_knots = np.array(
        [
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            in_d,
            in_d,
            in_d,
            in_d,
            in_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
            out_d,
        ]
    )
    f_eta = cs.interpolant("eta_bnd", "bspline", [xi_knots], eta_knots)

    return f_eta


if __name__ == "__main__":
    nlp_params = {
        "N": 100,
        "integration": "RK4",
        "verbose": True,
        "max_iter": 1000,
        "acceptable_tol": 1e-4,
    }

    # -------------------------------- Initialize -------------------------------- #

    # evaluate path
    f_ref, path = generate_path_functions()
    f_eta = get_bound()

    # robotic manipulator
    xi0 = 0.0
    eta0 = 0.0
    xif = 1.0
    etaf = 0.0

    p20 = spatial_to_cartesian(xi=xi0, eta=eta0, path=path)
    p10, theta10, theta20 = inverse_kinematics(p20)

    p2f = spatial_to_cartesian(xi=xif, eta=etaf, path=path)
    p1f, theta1f, theta2f = inverse_kinematics(p2f)

    x_start = np.array([xi0, eta0, 0.1, 0, theta10, theta20, 0, 0])
    x_end = np.array([xif, etaf, 0.1, 0, theta1f, theta2f, 0, 0])

    # ------------------------------------ NLP ----------------------------------- #

    N = nlp_params["N"]
    d_xi = xif / N

    # model
    f, f_xidot, f_cartesian = robotic_manipulator_ffullspatial(path)

    # decision variables
    nx = 8
    nu = 2
    x = cs.MX.sym("x", N, nx)
    u = cs.MX.sym("u", N - 1, nu)
    x_nlp = cs.vertcat(
        cs.reshape(x, x.size1() * x.size2(), 1), cs.reshape(u, u.size1() * u.size2(), 1)
    )  # [px_0,...,px_N,py_0,...py_N,...]

    # parameters
    # x0 = cs.SX.sym("x0", nx)
    # xf = cs.SX.sym("xf", nx)
    p_nlp = []  # cs.vertcat(x0, xf)

    # formulate NLP
    f_nlp = 0
    g_nlp = []
    lbg = []
    ubg = []

    for k in range(N):

        # initial and final states
        if k == 0:
            g_nlp = cs.vertcat(g_nlp, x[0, :].T - x_start)
            lbg = cs.vertcat(lbg, [0] * nx)
            ubg = cs.vertcat(ubg, [0] * nx)
        elif k == N - 1:
            g_nlp = cs.vertcat(g_nlp, x[N - 1, :3].T - x_end[:3])
            lbg = cs.vertcat(lbg, [0] * 3)
            ubg = cs.vertcat(ubg, [0] * 3)

        # corridor constraints
        # g_nlp = cs.vertcat(g_nlp, x[k, 1])
        # lbg = cs.vertcat(lbg, [-0.05])
        # ubg = cs.vertcat(ubg, [0.05])

        eta_max = f_eta(x[k, 0])
        g_nlp = cs.vertcat(g_nlp, cs.vertcat(-x[k, 1] + eta_max, x[k, 1] + eta_max))
        lbg = cs.vertcat(lbg, [0, 0])
        ubg = cs.vertcat(ubg, [cs.inf, cs.inf])

        # time law constraints
        xidot = f_xidot(x[k, :].T)
        # g_nlp = cs.vertcat(g_nlp, xidot)
        # lbg = cs.vertcat(lbg, [0.1])
        # ubg = cs.vertcat(ubg, [5])

        # path parameter constraints
        g_nlp = cs.vertcat(g_nlp, x[k, 0])
        lbg = cs.vertcat(lbg, [xi0])
        ubg = cs.vertcat(ubg, [xif])

        # joint velocity constraints
        g_nlp = cs.vertcat(g_nlp, x[k, 6:8].T)
        lbg = cs.vertcat(lbg, [-1, -1])
        ubg = cs.vertcat(ubg, [1, 1])

        if k < N - 1:
            # joint acceleration constraints
            g_nlp = cs.vertcat(g_nlp, u[k, :].T)
            lbg = cs.vertcat(lbg, [-5, -5])
            ubg = cs.vertcat(ubg, [5, 5])

            # velocity constraints
            v = f_cartesian(x=x[k, :], u=u[k, :])["v"]
            g_nlp = cs.vertcat(g_nlp, v)
            lbg = cs.vertcat(lbg, [-1, -1])
            ubg = cs.vertcat(ubg, [1, 1])

            # continuity
            if nlp_params["integration"] == "euler":
                x_next = x[k, :] + d_xi * f(x[k, :], u[k, :]).T
            elif nlp_params["integration"] == "RK4":
                x_next = RK4(x[k, :].T, u[k, :].T, d_xi, f).T

            g_nlp = cs.vertcat(g_nlp, (x[k + 1, :] - x_next).T)
            lbg = cs.vertcat(lbg, [0] * nx)
            ubg = cs.vertcat(ubg, [0] * nx)

            # cost function
            xidot_next = f_xidot(x[k + 1, :])
            dt = 2 * d_xi / (xidot_next + xidot)
            f_nlp += dt + 1e-5 * cs.sum1((u[k, :].T) ** 2)  # dt

    # Generate solver
    nlp_dict = {"x": x_nlp, "f": f_nlp, "g": g_nlp, "p": p_nlp}
    nlp_opts = {
        "ipopt.linear_solver": "mumps",
        "ipopt.sb": "yes",
        "ipopt.acceptable_tol": nlp_params["acceptable_tol"],
        "ipopt.max_iter": nlp_params["max_iter"],
        "ipopt.print_level": 5 if nlp_params["verbose"] else 0,
        "print_time": True,
    }
    nlp_solver = cs.nlpsol("corridor_nlp", "ipopt", nlp_dict, nlp_opts)

    solver = {"solver": nlp_solver, "lbg": lbg, "ubg": ubg}

    # ------------------------------ Initialization ------------------------------ #
    x_init = np.zeros((N, nx))
    u_init = np.zeros((N - 1, nu))
    xi_grid = np.linspace(xi0, xif, N)
    for k in range(N):
        p2 = path["gamma"](xi_grid[k])
        p1, theta1, theta2 = inverse_kinematics(p2)
        x_init[k] = np.array(
            [xi_grid[k], 0, 0.1, 0, float(theta1), float(theta2), 0, 0]
        )
        if k < N - 1:
            u_init[k] = np.array([0.0, 0.0])

    x0 = np.hstack([x_init.T.flatten(), u_init.T.flatten()])

    # --------------------------------- Solve NLP -------------------------------- #
    sol = solver["solver"](
        x0=x0,
        lbg=solver["lbg"],
        ubg=solver["ubg"],
        # p=np.concatenate([x_start, x_end]),
    )
    status = solver["solver"].stats()["success"]
    if not status:
        print("NLP solver failed")

    # x_sol = x0
    x_sol = sol["x"]

    # restructure output
    x = np.squeeze(x_sol)[: nx * N].reshape(nx, N).T
    u = np.squeeze(x_sol)[nx * N :].reshape(nu, N - 1).T
    xidot = np.zeros(N)

    # convert from spatial to cartesian coordinates and get other variables
    p1 = np.zeros((N, 2))
    p2 = np.zeros((N, 2))
    v = np.zeros((N, 2))
    a = np.zeros((N, 2))
    for k in range(N):
        p1[k], p2[k] = forward_kinematics(theta1=x[k, 4], theta2=x[k, 5])
        p2[k] = spatial_to_cartesian(xi=x[k, 0], eta=x[k, 1], path=path)
        xidot[k] = np.squeeze(f_xidot(x[k]))
        if k < N - 1:
            v[k] = np.squeeze(f_cartesian(x=x[k], u=u[k])["v"])
            a[k] = np.squeeze(f_cartesian(x=x[k], u=u[k])["a"])

    # post process data
    n_eval = 1000
    xi_eval = np.linspace(0, 1, n_eval)
    gamma_eval = np.zeros((n_eval, 2))
    eta_eval = np.zeros(n_eval)
    bnd1 = np.zeros((n_eval, 2))
    bnd2 = np.zeros((n_eval, 2))
    for k in range(n_eval):
        gamma_eval[k, :] = np.squeeze(path["gamma"](xi_eval[k]))
        eta_eval[k] = np.squeeze(f_eta(xi_eval[k]))
        bnd1[k, :] = gamma_eval[k] - eta_eval[k] * np.squeeze(
            path["e2"](xi=xi_eval[k])["e2"]
        )
        bnd2[k, :] = gamma_eval[k] + eta_eval[k] * np.squeeze(
            path["e2"](xi=xi_eval[k])["e2"]
        )

    dt = np.zeros(N - 1)
    eta_t_eval = np.zeros(N)
    for k in range(N):
        eta_t_eval[k] = np.squeeze(f_eta(x[k, 0]))

        if k < N - 1:
            dt[k] = 2 * d_xi / (xidot[k + 1] + xidot[k])
    t_sim = np.concatenate([[0], np.cumsum(dt)])

    # ------------------------------- Save results ------------------------------- #
    if False:
        save_pickle(
            path="/Users/jonarrizabalaga/2joint/paper/example3/data",
            file_name="example_3",
            data={
                "x": x,
                "u": u,
                "v": v,
                "p": [p1, p2],
                "t_sim": t_sim,
                "path": gamma_eval,
                "bounds": [eta_t_eval, bnd1, bnd2],
            },
        )

    # --------------------------------- Visualize -------------------------------- #

    fig, ax = plt.subplots()
    ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.1)
    # if case == 2:
    ax.plot(bnd1[:, 0], bnd1[:, 1], "k-", alpha=0.1)
    ax.plot(bnd2[:, 0], bnd2[:, 1], "k-", alpha=0.1)
    # ax.plot(p20[0], p20[1], "go", markersize=8)
    # ax.plot(p2f[0], p2f[1], "ro", markersize=8)
    for k in np.linspace(0, N - 1, num=3, dtype=int):
        ax = plot_two_link_manipulator(ax, joint1=p1[k], joint2=p2[k], color="gray")
    ax.scatter(
        p2[:-1, 0],
        p2[:-1, 1],
        marker=".",
        alpha=1.0,
        lw=0.5,
        cmap="turbo",
        c=np.linalg.norm(v[:-1, :], axis=1),
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig = plt.figure()
    plt.subplot(411)
    plt.plot(t_sim, x[:, 1])
    plt.plot(t_sim, eta_t_eval, "k--")
    plt.plot(t_sim, -eta_t_eval, "k--")
    plt.ylabel(r"$\eta$")
    plt.subplot(412)
    plt.plot(t_sim, v)
    plt.plot([t_sim[0], t_sim[-1]], [-1, -1], "k--")
    plt.plot([t_sim[0], t_sim[-1]], [1, 1], "k--")
    plt.ylabel(r"$v$")
    plt.subplot(413)
    plt.plot(t_sim, x[:, 6], "b", label=r"$\dot{\theta}_{1}$")
    plt.plot(t_sim, x[:, 7], "g", label=r"$\dot{\theta}_{2}$")
    plt.plot([t_sim[0], t_sim[-1]], [-1, -1], "k--")
    plt.plot([t_sim[0], t_sim[-1]], [1, 1], "k--")
    plt.ylabel(r"$\dot{\theta}$")
    plt.subplot(414)
    plt.plot([t_sim[0], t_sim[-1]], [-5, -5], "k--")
    plt.plot([t_sim[0], t_sim[-1]], [5, 5], "k--")
    plt.plot(t_sim[:-1], u[:, 0], "b", label=r"$\ddot{\theta}_{1}$")
    plt.plot(t_sim[:-1], u[:, 1], "g", label=r"$\ddot{\theta}_{2}$")
    plt.ylabel(r"$\ddot{\theta}$")

    plt.tight_layout()
    plt.show()
