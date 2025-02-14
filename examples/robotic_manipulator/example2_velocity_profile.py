# %%
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs

import pickle
import argparse
from sys import exit
from scipy.interpolate import CubicSpline

from examples.robotic_manipulator.src.model import (
    forward_kinematics,
    inverse_kinematics,
    forward_kinematics,
    robotic_manipulator_spatial,
    robotic_manipulator_fullspatial,
    spatial_to_cartesian,
    calculate_xidot,
)
from examples.robotic_manipulator.src.controller import (
    mpc_spatial_solver,
    mpc_fullspatial_solver,
)
from examples.robotic_manipulator.src.utils import (
    plot_two_link_manipulator,
    save_pickle,
)


def generate_path_functions(case):
    # define path
    xi = cs.SX.sym("xi")
    gamma = cs.vertcat(xi, 0.5 * cs.sin(2 * np.pi * xi) + 1)
    if case == 0:
        xi_dot = 0.15
    elif case == 1:
        xi_dot = 0.05
    elif case == 2:
        xi_dot = 0.12 + 0.02 * cs.sin(3 * 2 * np.pi * xi)
    elif case == 3:
        xi_dot = 0.08 + 0.02 * cs.cos(6 * 2 * np.pi * xi)
        # xi_dot = 0.05 - 0.07 * (xi - 0.5) ** 2

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
    f_xidot = cs.Function("f_xidot", [xi], [xi_dot])

    path = {
        "gamma": f_gamma,
        "e1": f_e1,
        "e2": f_e2,
        "sigma": f_sigma,
        "omega": f_omega,
        "xi_dot": f_xidot,
        "eta_min": -0.05,
        "eta_max": 0.05,
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
        xidot_ref[k] = np.squeeze(f_xidot(xi_eval[k]))
        theta1_ref[k] = theta1_ref_k
        theta2_ref[k] = theta2_ref_k

    xidot_spl = CubicSpline(xi_eval, xidot_ref)
    xi_spl = CubicSpline(xi_eval, xi_ref)
    eta_spl = CubicSpline(xi_eval, eta_ref)
    theta1_spl = CubicSpline(xi_eval, theta1_ref)
    theta2_spl = CubicSpline(xi_eval, theta2_ref)

    f_ref = lambda xi: np.array(
        [
            xi_spl(xi),
            eta_spl(xi),
            xidot_spl(xi),
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


if __name__ == "__main__":

    T = 2  # time horizion for MPC
    N = 20  # number of nodes for MPC

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=int,
        default=1,
        help="1: constant, 2: sinusoidal, 3: quadratic",
    )
    args = parser.parse_args()
    case = args.case

    # -------------------------------- Initialize -------------------------------- #
    # define path
    xi = cs.SX.sym("xi")
    f_ref, path = generate_path_functions(case=case)

    # evaluate path
    n_eval = 1000
    xi_eval = np.linspace(0, 1, n_eval)
    gamma_eval = np.zeros((n_eval, 2))
    bnd1 = np.zeros((n_eval, 2))
    bnd2 = np.zeros((n_eval, 2))
    for k in range(n_eval):
        gamma_eval[k, :] = np.squeeze(path["gamma"](xi_eval[k]))

    # robotic manipulator
    if case == 0:
        xi0 = 0.01
        eta0 = -0.3
    elif case == 1:
        xi0 = 0.1
        eta0 = 0.1
    elif case == 2:
        xi0 = 0.4
        eta0 = -0.2
    elif case == 3:
        xi0 = 0.55
        eta0 = 0.2
    p20 = spatial_to_cartesian(xi=xi0, eta=eta0, path=path)
    p10, theta10, theta20 = inverse_kinematics(p20)

    x0 = np.array([xi0, eta0, 0, 0, theta10, theta20, 0, 0])

    # ------------------------- test parametric functions ------------------------ #
    if False:
        xi_eval = np.linspace(0, 1, 1000)
        gamma_eval = np.zeros((len(xi_eval), 2))
        e1_eval = np.zeros((len(xi_eval), 2))
        e2_eval = np.zeros((len(xi_eval), 2))
        sigma_eval = np.zeros(len(xi_eval))
        omega_eval = np.zeros(len(xi_eval))
        for k in range(len(xi_eval)):
            gamma_eval[k, :] = np.squeeze(path["gamma"](xi_eval[k]))
            e1_eval[k, :] = np.squeeze(path["e1"](xi_eval[k]))
            e2_eval[k, :] = np.squeeze(path["e2"](xi_eval[k]))
            sigma_eval[k] = np.squeeze(path["sigma"](xi_eval[k]))
            omega_eval[k] = np.squeeze(path["omega"](xi_eval[k]))

        fig = plt.figure()

        ax = plt.subplot(121)
        scale = 0.1
        freq = 10
        ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.1)
        ax.plot(
            [gamma_eval[::freq, 0], gamma_eval[::freq, 0] + scale * e1_eval[::freq, 0]],
            [gamma_eval[::freq, 1], gamma_eval[::freq, 1] + scale * e1_eval[::freq, 1]],
            "r-",
            alpha=0.5,
        )
        ax.plot(
            [gamma_eval[::freq, 0], gamma_eval[::freq, 0] + scale * e2_eval[::freq, 0]],
            [gamma_eval[::freq, 1], gamma_eval[::freq, 1] + scale * e2_eval[::freq, 1]],
            "g-",
            alpha=0.5,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax = plt.subplot(222)
        ax.plot(xi_eval, sigma_eval)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\sigma$")

        ax = plt.subplot(224)
        ax.plot(xi_eval, omega_eval)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\omega$")

        plt.suptitle("Parametric paths")

    # ------------------------- test spatial plant model ------------------------- #
    if False:
        with open("/Users/jonarrizabalaga/2joint/data.pkl", "rb") as handle:
            data = pickle.load(handle)
        X = data["X"]
        U = data["U"]
        t_sim = data["t_sim"]

        # plt.plot(t_sim, U[:, 0])
        # plt.plot(t_sim, U[:, 1])

        model = robotic_manipulator_spatial(x0, path, func=True)

        N_test = t_sim.shape[0]
        x_test = np.zeros((N_test, 8))
        x_test[0] = x0
        for k in range(N_test - 1):
            x_dot = np.squeeze(model.f(x_test[k], U[k]))
            x_test[k + 1] = x_test[k] + x_dot * (t_sim[k + 1] - t_sim[k])

        p1 = x_test[:, :2]
        p2 = np.zeros((N_test, 2))
        for kk in range(N_test):
            p2[kk] = np.squeeze(path["gamma"](x_test[kk, 2])) + x_test[
                kk, 3
            ] * np.squeeze(path["e2"](x_test[kk, 2]))

        # plt.plot(p1[:, 0], p1[:, 1], "b-", alpha=0.3, lw=0.5)
        # plt.plot(p2[:, 0], p2[:, 1], "g.", alpha=1.0, lw=0.5)

        plt.subplot(421)
        plt.plot(t_sim, X[:-1, 0], "--")
        plt.plot(t_sim, p1[:, 0], "-")
        plt.subplot(423)
        plt.plot(t_sim, X[:-1, 1], "--")
        plt.plot(t_sim, p1[:, 1], "-")
        plt.subplot(425)
        plt.plot(t_sim, X[:-1, 2], "--")
        plt.plot(t_sim, p2[:, 0], "-")
        plt.subplot(427)
        plt.plot(t_sim, X[:-1, 3], "--")
        plt.plot(t_sim, p2[:, 1], "-")
        plt.subplot(222)
        plt.plot(t_sim, x_test[:, 2])
        plt.subplot(224)
        plt.plot(t_sim, x_test[:, 3])

    # %%
    # mpc solver
    mpc = mpc_fullspatial_solver(T=T, N=N, x0=x0, path=path)

    # -------------------------------- Simulation -------------------------------- #
    T_sim = 20
    dt = T / N
    t_sim = np.arange(0, T_sim, dt)
    N_sim = len(t_sim)
    X_ref = np.zeros((N_sim, 10))
    X = np.zeros((N_sim + 1, 8))
    X_pred = np.zeros((N_sim + 1, 8, N))
    U = np.zeros((N_sim, 2))

    X[0] = x0
    k = 0
    t_disturbance = 0

    # for kk in range(N):
    #     xi_kk = min(xi0 + (T / N * kk) / T_sim, 1)
    #     x0_k = np.hstack([xi_kk, x0[1:]])
    #     mpc.set(k, "x", X[k])

    while True:

        # set initial state
        mpc.set(0, "lbx", X[k])
        mpc.set(0, "ubx", X[k])

        # set reference
        for kk in range(N + 1):

            # get current progress variable
            # p2 = spatial_to_cartesian(xi=X[k, 2], eta=X[k, 3], path=path)
            # ind_gamma = np.argmin(np.sum((gamma_eval - p2) ** 2, axis=1))
            xi0 = X[k, 0]
            # xi_kk = min(xi0 + (T / N * kk) / T_sim, 1)

            xi_kk = min(xi0 + (T / N * kk) / T_sim, 1)

            # get reference
            p_ref_k = f_ref(xi_kk)

            # set reference in solver
            mpc.set(kk, "p", p_ref_k)

            # save reference
            if kk == 0 and k < N_sim:
                X_ref[k] = p_ref_k

        # solve
        status = mpc.solve()
        debug = 0
        if status != 0:
            if status == 1:
                print("acados returned status 1, failure")
                debug = 1
            elif status == 2:
                print("Warning: Acados returned status 2, max iter")
            else:
                print("acados returned status {}.".format(status))
                debug = 1
                break

        x = mpc.get(1, "x")
        u = mpc.get(0, "u")
        for kk in range(N):
            X_pred[k, :, kk] = mpc.get(kk, "x")

        # update states
        if k >= N_sim:
            break
        else:
            X[k + 1, :] = x
            U[k, :] = u
            k += 1

        # check if reached end and print status
        # if xi_kk >= xi_eval[-1]:  # ind_gamma == n_eval - 1:
        #     break
        # print("xi,xi_ref:", str(x[2]), str(xi0 + (T / N) / T_sim))
        print("xi = " + str(x[0]) + "/" + str(xi_eval[-1]))
        # print(x0 - mpc.get(0, "x"), "\n")

        if xi0 >= 0.99:
            X = X[: k + 1, :]
            U = U[:k, :]
            X_ref = X_ref[:k, :]
            X_pred = X_pred[: k + 1, :, :]
            N_sim = k
            t_sim = t_sim[:k]
            print("Finished!")
            break

        debug = 0
        if debug:
            # end effector from spatial to cartesian coordinates
            p1 = np.zeros((N, 2))
            p2 = np.zeros((N, 2))
            for kk in range(N):
                p1[kk], p2[kk] = forward_kinematics(
                    theta1=X_pred[k - 1, 4, kk], theta2=X_pred[k - 1, 5, kk]
                )
                # p2[kk] = spatial_to_cartesian(
                #     xi=X_pred[0, 0, kk], eta=X_pred[0, 1, kk], path=path
                # )

                # print(p1[kk], p2[kk])

            # plot horizion for debugging
            fig, ax = plt.subplots()
            ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.1)
            ax = plot_two_link_manipulator(ax, joint1=p1[0], joint2=p2[0])
            ax.plot(p2[:, 0], p2[:, 1], "g.", alpha=1.0, lw=0.5)
            plt.show()

            # mpc.print_statistics()
            # exit()
            # break

    # convert from spatial to cartesian coordinates and get other variables
    p1 = np.zeros((N_sim, 2))
    p2 = np.zeros((N_sim, 2))
    for k in range(N_sim):
        # p2[k] = spatial_to_cartesian(xi=X[k, 0], eta=X[k, 1], path=path)
        p1[k], p2[k] = forward_kinematics(theta1=X[k, 4], theta2=X[k, 5])

    # ------------------------------- Save results ------------------------------- #
    if False:
        save_pickle(
            path="/Users/jonarrizabalaga/2joint/paper/example2/data",
            file_name="case_" + str(case),
            data={
                "p": p2,
                "X": X,
                "U": U,
                "X_ref": X_ref,
                "t_sim": t_sim,
                "path": gamma_eval,
            },
        )

    # --------------------------------- Visualize -------------------------------- #
    t_sim = X[:-1, 0]

    # visualize
    fig, ax = plt.subplots()
    ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.1)
    ax.plot(bnd1[:, 0], bnd1[:, 1], "k-", alpha=0.1)
    ax.plot(bnd2[:, 0], bnd2[:, 1], "k-", alpha=0.1)
    for k in np.linspace(0, N_sim - 1, num=3, dtype=int):
        ax = plot_two_link_manipulator(ax, joint1=p1[k], joint2=p2[k])
    # ax.plot(X[:, 0], X[:, 1], "b-", alpha=0.3, lw=0.5)
    ax.plot(p2[:, 0], p2[:, 1], "g-", alpha=1.0, lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.figure()
    plt.subplot(311)
    plt.plot(t_sim, X_ref[:, 4], "b--", label=r"$\theta_{1,ref}$")
    plt.plot(t_sim, X[:-1, 4], "b", label=r"$\theta_{1}$")
    plt.plot(t_sim, X_ref[:, 5], "g--", label=r"$\theta_{2,ref}$")
    plt.plot(t_sim, X[:-1, 5], "g", label=r"$\theta_{2}$")
    plt.legend()
    plt.subplot(312)
    # plt.plot(t_sim, X_ref[:, 6], "b--", label=r"$\dot{\theta}_{1,ref}$")
    # plt.plot(t_sim, X_ref[:, 7], "g--", label=r"$\dot{\theta}_{2,ref}$")
    plt.plot(t_sim, X[:-1, 6], "b", label=r"$\dot{\theta}_{1}$")
    plt.plot(t_sim, X[:-1, 7], "g", label=r"$\dot{\theta}_{2}$")
    plt.legend()
    plt.subplot(313)
    # plt.plot(t_sim, X_ref[:, 8], "b--", label=r"$\dot{\theta}_{1,ref}$")
    # plt.plot(t_sim, X_ref[:, 9], "g--", label=r"$\dot{\theta}_{2,ref}$")
    plt.plot(t_sim, U[:, 0], "b", label=r"$\ddot{\theta}_{1}$")
    plt.plot(t_sim, U[:, 1], "g", label=r"$\ddot{\theta}_{2}$")
    plt.legend()
    plt.xlabel(r"$\xi$")
    plt.suptitle("Configuration space")

    plt.figure()
    plt.subplot(221)
    plt.plot(t_sim, X[:-1, 0])
    plt.ylabel(r"$\xi$")
    plt.subplot(222)
    plt.plot(t_sim, X[:-1, 1])
    plt.ylabel(r"$\eta$")
    plt.subplot(223)
    plt.plot(t_sim, X[:-1, 2])
    plt.plot(t_sim, X_ref[:, 2], "--")
    plt.ylabel(r"$\dot{\xi}$")
    plt.xlabel(r"$\xi$")
    plt.subplot(224)
    plt.plot(t_sim, X[:-1, 3])
    plt.ylabel(r"$\dot{\eta}$")
    plt.xlabel(r"$\xi$")
    plt.suptitle("Spatial coordinates")

    plt.show()
