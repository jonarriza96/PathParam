import numpy as np
import casadi as cs

from acados_template import AcadosModel

L1 = 1
L2 = 1

# ---------------------------------- Models ---------------------------------- #


def robotic_manipulator(x0):

    # states
    x1 = cs.SX.sym("x1")
    y1 = cs.SX.sym("y1")
    x2 = cs.SX.sym("x2")
    y2 = cs.SX.sym("y2")
    theta1 = cs.SX.sym("theta1")
    theta2 = cs.SX.sym("theta2")
    theta1_dot = cs.SX.sym("theta1_dot")
    theta2_dot = cs.SX.sym("theta2_dot")
    x = cs.vertcat(x1, y1, x2, y2, theta1, theta2, theta1_dot, theta2_dot)
    xdot = cs.SX.sym("xdot", x.size1())

    # inputs
    theta1_ddot = cs.SX.sym("theta1_ddot")
    theta2_ddot = cs.SX.sym("theta2_ddot")
    u = cs.vertcat(theta1_ddot, theta2_ddot)

    # equations of motion
    x1 = x[0]
    y1 = x[1]
    x2 = x[2]
    y2 = x[3]
    theta1 = x[4]
    theta2 = x[5]
    theta1_dot = x[6]
    theta2_dot = x[7]
    theta1_ddot = u[0]
    theta2_ddot = u[1]

    x1_dot = -L1 * theta1_dot * cs.sin(theta1)
    y1_dot = L1 * theta1_dot * cs.cos(theta1)
    x2_dot = -L1 * theta1_dot * cs.sin(theta1) - L2 * (
        theta1_dot + theta2_dot
    ) * cs.sin(theta1 + theta2)
    y2_dot = L1 * theta1_dot * cs.cos(theta1) + L2 * (theta1_dot + theta2_dot) * cs.cos(
        theta1 + theta2
    )
    f_expl = cs.vertcat(
        x1_dot, y1_dot, x2_dot, y2_dot, theta1_dot, theta2_dot, theta1_ddot, theta2_ddot
    )

    # create model
    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = cs.vertcat([])
    model.p = cs.SX.sym("x_ref", 10)
    model.x0 = x0
    model.name = "robotic_manipulator"

    return model


def robotic_manipulator_spatial(x0, path, func=False):

    # states
    x1 = cs.SX.sym("x1")
    y1 = cs.SX.sym("y1")
    xi = cs.SX.sym("xi")
    eta = cs.SX.sym("eta")
    theta1 = cs.SX.sym("theta1")
    theta2 = cs.SX.sym("theta2")
    theta1_dot = cs.SX.sym("theta1_dot")
    theta2_dot = cs.SX.sym("theta2_dot")
    x = cs.vertcat(x1, y1, xi, eta, theta1, theta2, theta1_dot, theta2_dot)
    xdot = cs.SX.sym("xdot", x.size1())

    # inputs
    theta1_ddot = cs.SX.sym("theta1_ddot")
    theta2_ddot = cs.SX.sym("theta2_ddot")
    u = cs.vertcat(theta1_ddot, theta2_ddot)

    # states
    x1 = x[0]
    y1 = x[1]
    xi = x[2]
    eta = x[3]
    theta1 = x[4]
    theta2 = x[5]
    theta1_dot = x[6]
    theta2_dot = x[7]
    theta1_ddot = u[0]
    theta2_ddot = u[1]

    # path variables
    e1 = path["e1"](xi)
    e2 = path["e2"](xi)
    sigma = path["sigma"](xi)
    omega = path["omega"](xi)

    # equations of motion
    v1 = cs.vertcat(-L1 * theta1_dot * cs.sin(theta1), L1 * theta1_dot * cs.cos(theta1))
    v2 = cs.vertcat(
        -L1 * theta1_dot * cs.sin(theta1)
        - L2 * (theta1_dot + theta2_dot) * cs.sin(theta1 + theta2),
        L1 * theta1_dot * cs.cos(theta1)
        + L2 * (theta1_dot + theta2_dot) * cs.cos(theta1 + theta2),
    )

    xi_dot = cs.dot(e1, v2) / (sigma - omega * eta)
    eta_dot = cs.dot(e2, v2)
    f_expl = cs.vertcat(
        v1,
        xi_dot,
        eta_dot,
        theta1_dot,
        theta2_dot,
        theta1_ddot,
        theta2_ddot,
    )
    f = cs.Function("f", [x, u], [f_expl])

    # create model
    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    if func:
        model.f = f
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = cs.vertcat([])
    model.p = cs.SX.sym("x_ref", 10)
    model.x0 = x0
    model.name = "robotic_manipulator"

    return model


def robotic_manipulator_fullspatial(x0, path, func=False):

    # states
    xi = cs.SX.sym("xi")
    eta = cs.SX.sym("eta")
    xi_dot = cs.SX.sym("xi_dot")
    eta_dot = cs.SX.sym("eta_dot")
    theta1 = cs.SX.sym("theta1")
    theta2 = cs.SX.sym("theta2")
    theta1_dot = cs.SX.sym("theta1_dot")
    theta2_dot = cs.SX.sym("theta2_dot")
    x = cs.vertcat(xi, eta, xi_dot, eta_dot, theta1, theta2, theta1_dot, theta2_dot)
    xdot = cs.SX.sym("xdot", x.size1())

    # inputs
    theta1_ddot = cs.SX.sym("theta1_ddot")
    theta2_ddot = cs.SX.sym("theta2_ddot")
    u = cs.vertcat(theta1_ddot, theta2_ddot)

    # path variables
    e1 = path["e1"](xi=xi)["e1"]
    e2 = path["e2"](xi=xi)["e2"]
    sigma = path["sigma"](xi=xi)["sigma"]
    omega = path["omega"](xi=xi)["omega"]

    e1_d = path["e1"](xi=xi)["e1_d"]
    e2_d = path["e2"](xi=xi)["e2_d"]
    sigma_d = path["sigma"](xi=xi)["sigma_d"]
    omega_d = path["omega"](xi=xi)["omega_d"]

    e1_dot = e1_d * xi_dot
    e2_dot = e2_d * xi_dot
    sigma_dot = sigma_d * xi_dot
    omega_dot = omega_d * xi_dot

    # equations of motion
    v2 = cs.vertcat(
        -L1 * theta1_dot * cs.sin(theta1)
        - L2 * (theta1_dot + theta2_dot) * cs.sin(theta1 + theta2),
        L1 * theta1_dot * cs.cos(theta1)
        + L2 * (theta1_dot + theta2_dot) * cs.cos(theta1 + theta2),
    )
    a2 = cs.vertcat(
        # x-component
        -L1 * (theta1_ddot * cs.sin(theta1) + theta1_dot**2 * cs.cos(theta1))
        - L2
        * (
            (theta1_ddot + theta2_ddot) * cs.sin(theta1 + theta2)
            + (theta1_dot + theta2_dot) ** 2 * cs.cos(theta1 + theta2)
        ),
        # y-component
        L1 * (theta1_ddot * cs.cos(theta1) - theta1_dot**2 * cs.sin(theta1))
        + L2
        * (
            (theta1_ddot + theta2_ddot) * cs.cos(theta1 + theta2)
            - (theta1_dot + theta2_dot) ** 2 * cs.sin(theta1 + theta2)
        ),
    )

    xi_ddot = (cs.dot(e1_dot, v2) + cs.dot(e1, a2)) / (sigma - omega * eta) - (
        cs.dot(e1, v2) / (sigma - omega * eta) ** 2
    ) * (sigma_dot - omega_dot * eta - omega * eta_dot)
    eta_ddot = cs.dot(e2_dot, v2) + cs.dot(e2, a2)

    f_expl = cs.vertcat(
        xi_dot,
        eta_dot,
        xi_ddot,
        eta_ddot,
        theta1_dot,
        theta2_dot,
        theta1_ddot,
        theta2_ddot,
    )
    f = cs.Function("f", [x, u], [f_expl])

    # create model
    model = AcadosModel()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    if func:
        model.f = f
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = cs.vertcat([])
    model.p = cs.SX.sym("x_ref", 10)
    model.x0 = x0
    model.name = "robotic_manipulator"

    return model


def robotic_manipulator_ffullspatial(path):

    # states
    xi = cs.SX.sym("xi")
    eta = cs.SX.sym("eta")
    xi_dot = cs.SX.sym("xi_dot")
    eta_dot = cs.SX.sym("eta_dot")
    theta1 = cs.SX.sym("theta1")
    theta2 = cs.SX.sym("theta2")
    theta1_dot = cs.SX.sym("theta1_dot")
    theta2_dot = cs.SX.sym("theta2_dot")
    x = cs.vertcat(xi, eta, xi_dot, eta_dot, theta1, theta2, theta1_dot, theta2_dot)
    xdot = cs.SX.sym("xdot", x.size1())

    # inputs
    theta1_ddot = cs.SX.sym("theta1_ddot")
    theta2_ddot = cs.SX.sym("theta2_ddot")
    u = cs.vertcat(theta1_ddot, theta2_ddot)

    # path variables
    e1 = path["e1"](xi=xi)["e1"]
    e2 = path["e2"](xi=xi)["e2"]
    sigma = path["sigma"](xi=xi)["sigma"]
    omega = path["omega"](xi=xi)["omega"]

    e1_d = path["e1"](xi=xi)["e1_d"]
    e2_d = path["e2"](xi=xi)["e2_d"]
    sigma_d = path["sigma"](xi=xi)["sigma_d"]
    omega_d = path["omega"](xi=xi)["omega_d"]

    e1_dot = e1_d * xi_dot
    e2_dot = e2_d * xi_dot
    sigma_dot = sigma_d * xi_dot
    omega_dot = omega_d * xi_dot

    # equations of motion
    v2 = cs.vertcat(
        -L1 * theta1_dot * cs.sin(theta1)
        - L2 * (theta1_dot + theta2_dot) * cs.sin(theta1 + theta2),
        L1 * theta1_dot * cs.cos(theta1)
        + L2 * (theta1_dot + theta2_dot) * cs.cos(theta1 + theta2),
    )
    a2 = cs.vertcat(
        # x-component
        -L1 * (theta1_ddot * cs.sin(theta1) + theta1_dot**2 * cs.cos(theta1))
        - L2
        * (
            (theta1_ddot + theta2_ddot) * cs.sin(theta1 + theta2)
            + (theta1_dot + theta2_dot) ** 2 * cs.cos(theta1 + theta2)
        ),
        # y-component
        L1 * (theta1_ddot * cs.cos(theta1) - theta1_dot**2 * cs.sin(theta1))
        + L2
        * (
            (theta1_ddot + theta2_ddot) * cs.cos(theta1 + theta2)
            - (theta1_dot + theta2_dot) ** 2 * cs.sin(theta1 + theta2)
        ),
    )

    xi_ddot = (cs.dot(e1_dot, v2) + cs.dot(e1, a2)) / (sigma - omega * eta) - (
        cs.dot(e1, v2) / (sigma - omega * eta) ** 2
    ) * (sigma_dot - omega_dot * eta - omega * eta_dot)
    eta_ddot = cs.dot(e2_dot, v2) + cs.dot(e2, a2)

    f_expl = cs.vertcat(
        xi_dot,
        eta_dot,
        xi_ddot,
        eta_ddot,
        theta1_dot,
        theta2_dot,
        theta1_ddot,
        theta2_ddot,
    )
    f = cs.Function("f", [x, u], [f_expl / xi_dot])
    f_xidot = cs.Function("f_xidot", [x], [xi_dot])
    f_cartesian = cs.Function("f_cartesian", [x, u], [v2, a2], ["x", "u"], ["v", "a"])

    return f, f_xidot, f_cartesian


# ------------------------------ Other functions ----------------------------- #


def inverse_kinematics(p):
    theta2 = np.arccos((p[0] ** 2 + p[1] ** 2 - L1**2 - L2**2) / (2 * L1 * L2))
    theta1 = np.arctan2(p[1], p[0]) - np.arctan2(
        L2 * np.sin(theta2), L1 + L2 * np.cos(theta2)
    )
    p1 = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    # theta2 = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0]))
    return p1, theta1, theta2


def forward_kinematics(theta1, theta2):
    p1 = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    p2 = p1 + np.array([L2 * np.cos(theta1 + theta2), L2 * np.sin(theta1 + theta2)])
    return p1, p2


def spatial_to_cartesian(xi, eta, path):
    return np.squeeze(path["gamma"](xi)) + eta * np.squeeze(path["e2"](xi=xi)["e2"])


def calculate_xidot(x, path):

    xi = x[2]
    eta = x[3]
    theta1 = x[4]
    theta2 = x[5]
    theta1_dot = x[6]
    theta2_dot = x[7]

    v2 = cs.vertcat(
        -L1 * theta1_dot * cs.sin(theta1)
        - L2 * (theta1_dot + theta2_dot) * cs.sin(theta1 + theta2),
        L1 * theta1_dot * cs.cos(theta1)
        + L2 * (theta1_dot + theta2_dot) * cs.cos(theta1 + theta2),
    )
    xidot = cs.dot(path["e1"](xi), v2) / (path["sigma"](xi) - path["omega"](xi) * eta)
    return xidot


def RK4(x, u, dt, f):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)

    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next
