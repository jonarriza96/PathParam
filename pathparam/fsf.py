import numpy as np
import casadi as cs


def FSF_function_from_sym(xi, p, v, a, j, s):
    e1 = v / cs.norm_2(v)
    e2 = cs.cross(v, (cs.cross(a, v))) / (cs.norm_2(v) * cs.norm_2(cs.cross(a, v)))
    e3 = cs.cross(v, a) / cs.norm_2(cs.cross(v, a))
    FSF = cs.horzcat(e1, e2, e3)

    e1d = cs.jacobian(e1, xi)
    e2d = cs.jacobian(e2, xi)
    e3d = cs.jacobian(e3, xi)
    FSFd = cs.horzcat(e1d, e2d, e3d)

    e1dd = cs.jacobian(e1d, xi)
    e2dd = cs.jacobian(e2d, xi)
    e3dd = cs.jacobian(e3d, xi)
    FSFdd = cs.horzcat(e1dd, e2dd, e3dd)

    omega1 = cs.dot(e2d, e3)
    omega2 = cs.dot(e3d, e1)
    omega3 = cs.dot(e1d, e2)
    omega = cs.vertcat(omega1, omega2, omega3)

    omegad = cs.jacobian(omega, xi)
    omegadd = cs.jacobian(omegad, xi)

    f_FSF = cs.Function(
        "f_FSF",
        [xi],
        [FSF, omega],
        ["xi"],
        ["FSF", "omega"],
    )

    f_FSF_derivatives = cs.Function(
        "f_FSF",
        [xi],
        [FSFd, FSFdd, omegad, omegadd],
        ["xi"],
        ["FSFd", "FSFdd", "omegad", "omegadd"],
    )

    return f_FSF, f_FSF_derivatives


def FSF_function(f_path):
    xi = cs.MX.sym("xi")
    p, v, a, j, s = f_path(xi)

    f_FSF, f_FSF_derivatives = FSF_function_from_sym(xi, p, v, a, j, s)

    return f_FSF, f_FSF_derivatives


def FSF_evaluate(f_p, xi, ff_FSF=None):
    n_eval = xi.shape[0]
    p_FSF = np.zeros((n_eval, 3))
    FSF = np.zeros((n_eval, 3, 3))
    FSFd = np.zeros((n_eval, 3, 3))
    FSFdd = np.zeros((n_eval, 3, 3))
    omega_FSF = np.zeros((n_eval, 3))
    omegad_FSF = np.zeros((n_eval, 3))
    omegadd_FSF = np.zeros((n_eval, 3))
    if ff_FSF is None:
        f_FSF, f_FSF_derivatives = FSF_function(f_path=f_p)
    else:
        f_FSF = ff_FSF["FSF"]
        f_FSF_derivatives = ff_FSF["FSFd"]

    for i in range(n_eval):
        p = f_p(xi=xi[i])["p"]
        FSFi, omegai = f_FSF(xi[i])
        FSFid, FSFidd, omegaid, omegaidd = f_FSF_derivatives(xi[i])

        p_FSF[i, :] = np.squeeze(p)
        FSF[i, :, :] = np.squeeze(FSFi)
        FSFd[i, :, :] = np.squeeze(FSFid)
        FSFdd[i, :, :] = np.squeeze(FSFidd)
        omega_FSF[i] = np.squeeze(omegai)
        omegad_FSF[i] = np.squeeze(omegaid)
        omegadd_FSF[i] = np.squeeze(omegaidd)

    return p_FSF, FSF, FSFd, FSFdd, omega_FSF, omegad_FSF, omegadd_FSF
