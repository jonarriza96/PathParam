import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

from scipy.linalg import expm
from pathparam.utils import closest_to_A_perpendicular_to_B


def tangent_function_from_sym(xi, dr):
    e1 = dr / cs.norm_2(dr)
    e1_d = cs.jacobian(e1, xi)
    e1_dd = cs.jacobian(e1_d, xi)
    e1_ddd = cs.jacobian(e1_dd, xi)

    f = cs.Function(
        "f_tangent",
        [xi],
        [e1, e1_d, e1_dd, e1_ddd],
        ["xi"],
        ["e1", "e1_d", "e1_dd", "e1_ddd"],
    )

    return f


def tangent_function(f_path):
    xi = cs.MX.sym("xi")
    r, dr, ddr, dddr, ddddr = f_path(xi)
    f = tangent_function_from_sym(xi=xi, dr=dr)
    return f


def vec_to_skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def initial_frame(e10, e3_des):
    e30 = closest_to_A_perpendicular_to_B(A=np.array([0, 0, 1]), B=e10)
    e30 = e30 / np.linalg.norm(e30)
    e20 = np.cross(e30, e10)
    PTF0 = np.vstack((e10, e20, e30)).T
    return PTF0


def omega_components(R, e1d):
    "Inspiration from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7782312"
    e1 = R[:, 0]
    e2 = R[:, 1]
    e3 = R[:, 2]

    X2 = -e1d.dot(e3)  # -k2
    X3 = e1d.dot(e2)  # k1
    return np.array([0, X2, X3])


def omegad_components(R, Rd, e1dd):
    e1 = R[:, 0]
    e2 = R[:, 1]
    e3 = R[:, 2]

    e1d = Rd[:, 0]
    e2d = Rd[:, 1]
    e3d = Rd[:, 2]

    X2d = -(e1dd.dot(e3) + e1d.dot(e3d))
    X3d = e1dd.dot(e2) + e1d.dot(e2d)
    return np.array([0, X2d, X3d])


def omegadd_components(R, Rd, Rdd, e1ddd):
    e1 = R[:, 0]
    e2 = R[:, 1]
    e3 = R[:, 2]

    e1d = Rd[:, 0]
    e2d = Rd[:, 1]
    e3d = Rd[:, 2]

    e1dd = Rdd[:, 0]
    e2dd = Rdd[:, 1]
    e3dd = Rdd[:, 2]

    X2dd = -(e1ddd.dot(e3) + e1dd.dot(e3d) + e1dd.dot(e3d) + e1d.dot(e3dd))
    X3dd = e1ddd.dot(e2) + e1dd.dot(e2d) + e1dd.dot(e2d) + e1d.dot(e2dd)
    return np.array([0, X2dd, X3dd])


def calculate_omega(omega_comp, R):
    e1 = R[:, 0]
    e2 = R[:, 1]
    e3 = R[:, 2]
    omega = omega_comp[0] * e1 + omega_comp[1] * e2 + omega_comp[2] * e3
    return omega


def calculate_omegad(omega_comp, omegad_comp, R, Rd):
    e1 = R[:, 0]
    e2 = R[:, 1]
    e3 = R[:, 2]
    e1d = Rd[:, 0]
    e2d = Rd[:, 1]
    e3d = Rd[:, 2]
    omegad = (
        omegad_comp[0] * e1
        + omega_comp[0] * e1d
        + omegad_comp[1] * e2
        + omega_comp[1] * e2d
        + omegad_comp[2] * e3
        + omega_comp[2] * e3d
    )
    return omegad


def omega_components_to_skew(omega_comp, R):
    omega = calculate_omega(omega_comp=omega_comp, R=R)
    skew_omega = vec_to_skew(omega)
    return skew_omega


def omegad_components_to_skew(omega_comp, omegad_comp, R, Rd):
    omegad = calculate_omegad(
        omega_comp=omega_comp, omegad_comp=omegad_comp, R=R, Rd=Rd
    )
    skew_omegad = vec_to_skew(omegad)
    return skew_omegad


def ptf_moving_frame(f_p, f_e1, PTF0, xi):
    n_eval = xi.shape[0]

    # Initial state
    p_PTF = np.zeros((n_eval, 3))
    PTF = np.zeros((n_eval, 3, 3))
    omega_comp = np.zeros((n_eval, 3))
    p_PTF[0, :] = np.squeeze(f_p(xi=xi[0])["p"])
    PTF[0] = PTF0
    omega_comp[0, :] = omega_components(e1d=np.squeeze(f_e1(xi=xi[0])["e1_d"]), R=PTF0)

    # Integrate
    for i in range(n_eval - 1):

        # 1- MOVING FRAME
        omega_skew = omega_components_to_skew(omega_comp=omega_comp[i], R=PTF[i])
        d_xi = xi[i + 1] - xi[i]
        PTF[i + 1] = expm(omega_skew * d_xi) @ PTF[i]
        # PTF[i + 1] = PTF[i] + (omega_skew @ PTF[i]) * d_xi
        # PTF[i + 1] = PTF[i] + (PTF[i] @ vec_to_skew(omega_comp[i])) * d_xi

        # 2- POSITION
        p_PTF[i + 1, :] = np.squeeze(f_p(xi=xi[i + 1])["p"])

        # 3- ANGULAR VELOCITY
        omega_comp[i + 1, :] = omega_components(
            R=PTF[i + 1], e1d=np.squeeze(f_e1(xi=xi[i + 1])["e1_d"])
        )

    return xi, p_PTF, PTF, omega_comp


def ptf_moving_frame_derivatives(f_e1, PTF, omega_comp, xi):
    n_eval = xi.shape[0]
    omegad_comp = np.zeros((n_eval, 3))
    omegadd_comp = np.zeros((n_eval, 3))
    PTFd = np.zeros((n_eval, 3, 3))
    PTFdd = np.zeros((n_eval, 3, 3))
    for i in range(n_eval):

        # PTFd
        omega_skew = omega_components_to_skew(omega_comp=omega_comp[i], R=PTF[i])
        PTFd[i, :, :] = omega_skew @ PTF[i]

        # angular acc
        omegad_comp[i, :] = omegad_components(
            R=PTF[i], Rd=PTFd[i], e1dd=np.squeeze(f_e1(xi=xi[i])["e1_dd"])
        )

        # PTFdd
        omegad_skew = omegad_components_to_skew(
            omega_comp=omega_comp[i], omegad_comp=omegad_comp[i], R=PTF[i], Rd=PTFd[i]
        )
        PTFdd[i, :, :] = omegad_skew @ PTF[i] + omega_skew @ PTFd[i]

        # angular jerk
        omegadd_comp[i, :] = omegadd_components(
            R=PTF[i],
            Rd=PTFd[i],
            Rdd=PTFdd[i],
            e1ddd=np.squeeze(f_e1(xi=xi[i])["e1_ddd"]),
        )

    return omegad_comp, omegadd_comp, PTFd, PTFdd


def PTF_evaluate(f_p, xi, f_e1=None):
    # Tangent function
    if f_e1 is None:
        f_e1 = tangent_function(f_path=f_p)

    # Define initial moving frame
    PTF0 = initial_frame(
        e10=np.squeeze(f_e1(xi=xi[0])["e1"]), e3_des=np.array([0, 0, 1])
    )

    # Compute moving frame
    xi, p_PTF, PTF, omega_PTF = ptf_moving_frame(f_p=f_p, f_e1=f_e1, PTF0=PTF0, xi=xi)

    # Compute higher derivatives
    omegad_PTF, omegadd_PTF, PTFd, PTFdd = ptf_moving_frame_derivatives(
        f_e1=f_e1, PTF=PTF, omega_comp=omega_PTF, xi=xi
    )

    return p_PTF, PTF, PTFd, PTFdd, omega_PTF, omegad_PTF, omegadd_PTF
