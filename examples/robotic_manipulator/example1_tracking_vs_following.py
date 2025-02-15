import numpy as np
import matplotlib.pyplot as plt
import casadi as cs

from examples.robotic_manipulator.src.model import inverse_kinematics
from examples.robotic_manipulator.src.controller import mpc_solver
from examples.robotic_manipulator.src.utils import (
    plot_two_link_manipulator,
    save_pickle,
)

from scipy.interpolate import CubicSpline

import pickle

import argparse


def generate_reference_function():
    # define path
    xi = cs.SX.sym("xi")
    gamma = cs.vertcat(xi, 0.5 * cs.sin(2 * np.pi * xi) + 1)
    f_gamma = cs.Function("f_gamma", [xi], [gamma])

    # evaluate path
    n_eval = 1000
    xi_eval = np.linspace(0, 1, n_eval)
    gamma_eval = np.zeros((n_eval, 2))
    p1x_ref = np.zeros(n_eval)
    p1y_ref = np.zeros(n_eval)
    p2x_ref = np.zeros(n_eval)
    p2y_ref = np.zeros(n_eval)
    theta1_ref = np.zeros(n_eval)
    theta2_ref = np.zeros(n_eval)
    for k in range(n_eval):
        gamma_eval[k, :] = np.squeeze(f_gamma(xi_eval[k]))

        p2_ref_k = np.squeeze(f_gamma(xi_eval[k]))
        p1_ref_k, theta1_ref_k, theta2_ref_k = inverse_kinematics(p2_ref_k)

        p1x_ref[k] = p1_ref_k[0]
        p1y_ref[k] = p1_ref_k[1]
        p2x_ref[k] = p2_ref_k[0]
        p2y_ref[k] = p2_ref_k[1]
        theta1_ref[k] = theta1_ref_k
        theta2_ref[k] = theta2_ref_k

    p1x_spl = CubicSpline(xi_eval, p1x_ref)
    p1y_spl = CubicSpline(xi_eval, p1y_ref)
    p2x_spl = CubicSpline(xi_eval, p2x_ref)
    p2y_spl = CubicSpline(xi_eval, p2y_ref)
    theta1_spl = CubicSpline(xi_eval, theta1_ref)
    theta2_spl = CubicSpline(xi_eval, theta2_ref)

    f_ref = lambda xi: np.array(
        [
            p1x_spl(xi),
            p1y_spl(xi),
            p2x_spl(xi),
            p2y_spl(xi),
            theta1_spl(xi),
            theta2_spl(xi),
            theta1_spl(xi, 1),
            theta2_spl(xi, 1),
            theta1_spl(xi, 2),
            theta2_spl(xi, 2),
        ]
    ).T

    return f_ref


if __name__ == "__main__":

    T = 2  # time horizion for MPC
    N = 20  # number of nodes for MPC

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default="t",
        help="t: tracking, f: following",
    )
    parser.add_argument(
        "--d",
        action="store_true",
        help="Introduces disturbance that mimics the robot blockage",
    )
    args = parser.parse_args()
    case = args.case
    disturbance = args.d

    # -------------------------------- Initialize -------------------------------- #
    # define path
    xi = cs.SX.sym("xi")
    gamma = cs.vertcat(xi, 0.5 * cs.sin(2 * np.pi * xi) + 1)
    f_gamma = cs.Function("f_gamma", [xi], [gamma])

    # evaluate path
    n_eval = 1000
    xi_eval = np.linspace(0, 1, n_eval)
    gamma_eval = np.zeros((n_eval, 2))
    for k in range(n_eval):
        gamma_eval[k, :] = np.squeeze(f_gamma(xi_eval[k]))
    f_ref = generate_reference_function()

    # robotic manipulator
    p20 = np.squeeze(f_gamma(0))  # + np.array([0.0, -0.1])
    p10, theta10, theta20 = inverse_kinematics(p20)
    x0 = np.array([p10[0], p10[1], p20[0], p20[1], theta10, theta20, 0, 0])

    # mpc solver
    mpc = mpc_solver(T=T, N=N, x0=x0)

    # -------------------------------- Simulation -------------------------------- #
    T_sim = 20
    dt = T / N
    t_sim = np.arange(0, T_sim, dt)
    N_sim = len(t_sim)
    X_ref = np.zeros((N_sim, 10))
    xi_ref = np.zeros(N_sim)
    X = np.zeros((N_sim + 1, 8))
    X_pred = np.zeros((N_sim + 1, 8, N))
    U = np.zeros((N_sim, 2))

    X[0] = x0
    k = 0
    t_disturbance = 0
    while True:

        # set initial state
        mpc.set(0, "lbx", X[k])
        mpc.set(0, "ubx", X[k])

        # set reference
        for kk in range(N + 1):

            # get current progress variable
            if case == "t":  # tracking
                xi_kk = (t_sim[k] + T / N * kk) / T_sim
            elif case == "f":  # following
                ind_gamma = np.argmin(np.sum((gamma_eval - X[k, 2:4]) ** 2, axis=1))
                xi_kk = min(xi_eval[ind_gamma] + (T / N * kk) / T_sim, 1)

            # get reference
            p_ref_k = f_ref(xi_kk)

            # set reference in solver
            mpc.set(kk, "p", p_ref_k)

            # save reference
            if kk == 0 and k < N_sim:
                X_ref[k] = p_ref_k

        # solve
        mpc.solve()
        x = mpc.get(1, "x")
        u = mpc.get(0, "u")
        for kk in range(N):
            X_pred[k, :, kk] = mpc.get(kk, "x")

        # mimic robot blockage
        if disturbance and (0.25 <= xi_kk and t_disturbance <= 3):  # blocked
            t_disturbance += dt
            x_update = X[k]
            prefix = "--- Blocked"
        else:  # not blocked
            x_update = x
            prefix = ""
        # prefix = ""

        # update states
        if k >= N_sim:
            break
        else:
            X[k + 1, :] = x_update
            U[k, :] = u
            xi_ref[k] = xi_kk
            k += 1

        # check if reached end and print status
        if case == "t":
            if k == N_sim:
                break
            print("t = " + str(t_sim[k]) + "/" + str(t_sim[-1]) + " s" + prefix)
        elif case == "f":
            if ind_gamma == n_eval - 1:
                break
            print("xi = " + str(xi_eval[ind_gamma]) + "/" + str(xi_eval[-1]) + prefix)
        # print(x0 - mpc.get(0, "x"), "\n")

        # if t_sim[k] >= 5:
        #     break
        # from sys import exit

        # exit()

        # plot horizion for debugging
        # fig, ax = plt.subplots()
        # ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.1)
        # k = 0
        # ax = plot_two_link_manipulator(ax, joint1=X[k, 0:2], joint2=X[k, 2:4])
        # ax.plot(X_pred[k, 2, :], X_pred[k, 3, :], "g.", alpha=1.0, lw=0.5)
        # plt.show()

    # ------------------------------- Save results ------------------------------- #
    if False:
        save_pickle(
            path="/Users/jonarrizabalaga/PathParam/examples/robotic_manipulator/paper/example1/data",
            file_name="tracking_disturbance",
            data={
                "X": X,
                "U": U,
                "X_ref": X_ref,
                "t_sim": t_sim,
                "path": gamma_eval,
                "xi_ref": xi_ref,
            },
        )

    # --------------------------------- Visualize -------------------------------- #

    # visualize
    fig, ax = plt.subplots()
    ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.1)
    for k in np.linspace(0, N_sim - 1, num=3, dtype=int):
        ax = plot_two_link_manipulator(ax, joint1=X[k, 0:2], joint2=X[k, 2:4])
    # ax.plot(X[:, 0], X[:, 1], "b-", alpha=0.3, lw=0.5)
    ax.plot(X[:, 2], X[:, 3], "-", alpha=1.0, lw=0.5)
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
    plt.xlabel("Time [s]")
    plt.show()
