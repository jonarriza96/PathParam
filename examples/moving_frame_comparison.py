import argparse
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from pathparam.ptf import PTF_evaluate
from pathparam.fsf import FSF_evaluate

from pathparam.visualize import visualize_moving_frame, visualize_angular_velocity


def paths():

    xi = cs.SX.sym("xi")
    sinu = cs.vertcat(xi, cs.sin(2 * np.pi * xi), 0)
    exampl1 = cs.vertcat(
        1.5 * cs.sin(7.2 * xi), cs.cos(9 * xi), cs.exp(cs.cos(1.8 * xi))
    )
    helix = cs.vertcat(
        0.5 * cs.sin(3 * 2 * np.pi * xi), 0.5 * cs.cos(3 * 2 * np.pi * xi), 2 * xi
    )
    c4 = cs.vertcat(xi, 0.5 * xi**3 + 0.1 * xi**2 - 1 * xi + 0.6, 0)
    exampl2 = cs.vertcat(
        (0.6 + 0.3 * cs.cos(xi)) * cs.cos(2 * xi),
        (0.6 + 0.3 * cs.cos(xi)) * cs.sin(2 * xi),
        0.3 * cs.sin(7 * xi),
    )
    return xi, [sinu, helix, exampl2]


def path_function(path_parameter, equation):

    xi = path_parameter
    r = equation
    dr = cs.jacobian(r, xi)  # first derivative of analytical curve
    ddr = cs.jacobian(dr, xi)  # second derivative of analytical curve
    dddr = cs.jacobian(ddr, xi)  # third derivative of analytical curve
    ddddr = cs.jacobian(dddr, xi)  # fourth derivative of analytical curve

    f_p = cs.Function(
        "f_p", [xi], [r, dr, ddr, dddr, ddddr], ["xi"], ["p", "v", "a", "j", "s"]
    )

    no_path = {"f_p": f_p}  # , "f_v": v, "f_a": a}

    return no_path


if __name__ == "__main__":

    # User parameters
    view = True
    d_xi = 0.001  # step size in integration
    xi_eval = 1.0

    # Define reference paths
    path_parameter, equations = paths()
    n_paths = len(equations)

    # Perpare visualization
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["r", "g", "b"])
    axs1 = plt.figure().subplots(n_paths, 2, subplot_kw={"projection": "3d"})
    axs2 = plt.figure().subplots(2 * n_paths, 2)

    for i in range(n_paths):
        # ---------------------------------------------------------------------------- #
        #                             Path parameterization                            #
        # ---------------------------------------------------------------------------- #
        # Progress variable grid
        xi = np.arange(0, xi_eval + d_xi, d_xi)

        # Get nominal path fynction
        path_f = path_function(path_parameter=path_parameter, equation=equations[i])

        # PTF
        p_PTF, PTF, PTFd, PTFdd, omega_PTF, omegad_PTF, omegadd_PTF = PTF_evaluate(
            f_p=path_f["f_p"], xi=xi
        )

        # FSF
        p_FSF, FSF, FSFd, FSFdd, omega_FSF, omegad_FSF, omegadd_FSF = FSF_evaluate(
            f_p=path_f["f_p"], xi=xi
        )

        # ---------------------------------------------------------------------------- #
        #                                   Visualize                                  #
        # ---------------------------------------------------------------------------- #
        if view:
            if i == 0:
                titles = ["PTF", "FSF", "ERF"]
            else:
                titles = [None, None, None]

            if i == n_paths - 1:
                xlabel = [r"$\xi$"]
            else:
                xlabel = [None]

            # moving frame 3d comparison
            visualize_moving_frame(p=p_PTF, R=PTF, title=titles[0], ax=axs1[i][0])
            visualize_moving_frame(p=p_FSF, R=FSF, title=titles[1], ax=axs1[i][1])

            # angular velocity comparison
            visualize_angular_velocity(
                xi=xi,
                omega=omega_PTF,
                alpha=omegad_PTF,
                j=omegadd_PTF,
                title=titles[0],
                ax=[axs2[2 * i][0], axs2[2 * i + 1][0]],
                ylabel=[r"$\omega^{\Gamma}$", r"$j^{\Gamma}$"],
                xlabel=xlabel,
            )
            visualize_angular_velocity(
                xi=xi,
                omega=omega_FSF,
                alpha=omegad_FSF,
                j=omegadd_FSF,
                title=titles[1],
                ax=[axs2[2 * i][1], axs2[2 * i + 1][1]],
                xlabel=xlabel,
            )

    plt.show()
