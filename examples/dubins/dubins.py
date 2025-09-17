# %%
import math
import sys
import numpy as np
import casadi as cs

import matplotlib.pyplot as plt
from math import sin, cos, atan2, sqrt, pi

from pathparam.visualize import plot_frames
from examples.robotic_manipulator.src.utils import save_pickle


# -------------------------
# Helper functions
# -------------------------
def mod2pi(theta):
    """Normalize angle to [0, 2π)"""
    return (theta + 2 * pi) % (2 * pi)


def atan2_to_pi(y, x):
    """Atan2 that returns angle in [0, 2π)"""
    return mod2pi(atan2(y, x))


def polar(x, y):
    return sqrt(x * x + y * y), atan2(y, x)


def transform_to_local(start, goal, rho=1.0):
    """Transform goal pose into local frame at start with heading along +x"""
    x0, y0, th0 = start
    x1, y1, th1 = goal
    dx = x1 - x0
    dy = y1 - y0
    x = (cos(-th0) * dx - sin(-th0) * dy) / rho
    y = (sin(-th0) * dx + cos(-th0) * dy) / rho
    th = mod2pi(th1 - th0)
    return x, y, th


# -------------------------
# Candidate path formulas (corrected)
# -------------------------
def dubins_LSL(alpha, beta, d):
    """Left-Straight-Left path"""
    sa = sin(alpha)
    sb = sin(beta)
    ca = cos(alpha)
    cb = cos(beta)
    c_ab = cos(alpha - beta)

    # Standard LSL formula
    tmp1 = d + sa - sb
    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)

    if p_sq < 0:
        return None

    p = sqrt(p_sq)
    tmp2 = atan2(cb - ca, tmp1)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q


def dubins_RSR(alpha, beta, d):
    """Right-Straight-Right path"""
    tmp1 = d - sin(alpha) + sin(beta)
    p_sq = 2 + d * d - 2 * cos(alpha - beta) + 2 * d * (-sin(alpha) + sin(beta))
    if p_sq < 0:
        return None
    p = sqrt(p_sq)
    tmp2 = atan2((cos(alpha) - cos(beta)), tmp1)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(-beta + tmp2)
    return t, p, q


def dubins_LSR(alpha, beta, d):
    """Left-Straight-Right path"""
    p_sq = d * d + 2 * cos(alpha - beta) + 2 * d * (sin(alpha) + sin(beta)) - 2
    if p_sq < 0:
        return None
    p = sqrt(p_sq)
    tmp = atan2((-cos(alpha) - cos(beta)), (d + sin(alpha) + sin(beta))) - atan2(
        -2.0, p
    )
    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q


def dubins_RSL(alpha, beta, d):
    """Right-Straight-Left path"""
    p_sq = d * d + 2 * cos(alpha - beta) + 2 * d * (-sin(alpha) - sin(beta)) - 2
    if p_sq < 0:
        return None
    p = sqrt(p_sq)
    tmp = atan2((cos(alpha) + cos(beta)), (d - sin(alpha) - sin(beta))) - atan2(2.0, p)
    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return t, p, q


def dubins_RLR(alpha, beta, d):
    """Right-Left-Right path"""
    tmp = (6.0 - d * d + 2 * cos(alpha - beta) + 2 * d * (sin(alpha) - sin(beta))) / 8.0
    if abs(tmp) > 1:
        return None
    p = mod2pi(2 * pi - math.acos(tmp))
    t = mod2pi(
        alpha - atan2((cos(alpha) - cos(beta)), d - sin(alpha) + sin(beta)) + p / 2.0
    )
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def dubins_LRL(alpha, beta, d):
    """Left-Right-Left path"""
    tmp = (
        6.0 - d * d + 2 * cos(alpha - beta) + 2 * d * (-sin(alpha) + sin(beta))
    ) / 8.0
    if abs(tmp) > 1:
        return None
    p = mod2pi(2 * pi - math.acos(tmp))
    t = mod2pi(
        -alpha - atan2((cos(alpha) - cos(beta)), d + sin(alpha) - sin(beta)) + p / 2.0
    )
    q = mod2pi(beta - alpha - t + p)
    return t, p, q


# -------------------------
# Main solver
# -------------------------
def dubins_shortest_path(start, goal, rho=1.0):
    """Find the shortest Dubins path between two poses"""
    x, y, th = transform_to_local(start, goal, rho=rho)
    d, theta = polar(x, y)
    d = max(d, 1e-10)
    alpha = mod2pi(atan2(y, x))
    beta = mod2pi(th - alpha)

    candidates = {}
    for name, fn in [
        ("LSL", dubins_LSL),
        ("RSR", dubins_RSR),
        ("LSR", dubins_LSR),
        ("RSL", dubins_RSL),
        ("RLR", dubins_RLR),
        ("LRL", dubins_LRL),
    ]:
        res = fn(alpha, beta, d)
        if res:
            candidates[name] = res

    best = None
    best_path = None
    for k, (t, p, q) in candidates.items():
        L = (t + p + q) * rho
        if best is None or L < best:
            best = L
            best_path = (k, t * rho, p * rho, q * rho)
    return best, best_path


# -------------------------
# Path sampling for plotting
# -------------------------
def sample_dubins(start, path, rho=1.0, step=0.05):
    """Sample points along a Dubins path for visualization"""
    typ, t, p, q = path
    seq = []
    if typ == "LSL":
        seq = [("L", t), ("S", p), ("L", q)]
    if typ == "RSR":
        seq = [("R", t), ("S", p), ("R", q)]
    if typ == "LSR":
        seq = [("L", t), ("S", p), ("R", q)]
    if typ == "RSL":
        seq = [("R", t), ("S", p), ("L", q)]
    if typ == "RLR":
        seq = [("R", t), ("L", p), ("R", q)]
    if typ == "LRL":
        seq = [("L", t), ("R", p), ("L", q)]

    x, y, th = start
    pts = [(x, y)]
    ths = [th]

    for seg, length in seq:
        if seg == "S":
            # Straight segment - sample intermediate points
            n_steps = max(1, int(length / step))
            for i in range(1, n_steps + 1):
                s = (i / n_steps) * length
                nx = x + s * cos(th)
                ny = y + s * sin(th)
                pts.append((nx, ny))
            x = x + length * cos(th)
            y = y + length * sin(th)
            ths.append(th)
        else:
            # Curved segment - sample intermediate points
            # Note: length is already arc length (t*rho), so divide by rho to get angle
            ang_total = length / rho
            n_steps = max(1, int(abs(ang_total) / (step / rho)))

            if seg == "L":
                # Left turn
                cx, cy = x - rho * sin(th), y + rho * cos(th)
                for i in range(1, n_steps + 1):
                    ang = (i / n_steps) * ang_total
                    th_new = th + ang
                    nx = cx + rho * sin(th_new)
                    ny = cy - rho * cos(th_new)
                    pts.append((nx, ny))
                th += ang_total
                x, y = cx + rho * sin(th), cy - rho * cos(th)
                ths.append(th)
            else:
                # Right turn
                cx, cy = x + rho * sin(th), y - rho * cos(th)
                for i in range(1, n_steps + 1):
                    ang = (i / n_steps) * ang_total
                    th_new = th - ang
                    nx = cx - rho * sin(th_new)
                    ny = cy + rho * cos(th_new)
                    pts.append((nx, ny))
                th -= ang_total
                x, y = cx - rho * sin(th), cy + rho * cos(th)
                ths.append(th)
    return pts, ths


def compute_theta_dot_profile(path, rho=1.0, step=0.05):
    """
    Compute bang-bang theta_dot profile for a Dubins path

    Args:
        path: Dubins path tuple (path_type, t, p, q) from dubins_shortest_path
        rho: turning radius
        step: sampling step size

    Returns:
        arc_lengths: array of arc length parameters
        theta_dot_profile: corresponding angular velocities
    """
    path_type, t, p, q = path

    # Define the sequence of maneuvers
    if path_type == "LSL":
        seq = [("L", t), ("S", p), ("L", q)]
    elif path_type == "RSR":
        seq = [("R", t), ("S", p), ("R", q)]
    elif path_type == "LSR":
        seq = [("L", t), ("S", p), ("R", q)]
    elif path_type == "RSL":
        seq = [("R", t), ("S", p), ("L", q)]
    elif path_type == "RLR":
        seq = [("R", t), ("L", p), ("R", q)]
    elif path_type == "LRL":
        seq = [("L", t), ("R", p), ("L", q)]

    # Compute total length
    total_length = t + p + q

    # Sample points along the path
    n_points = max(10, int(total_length / step))
    arc_lengths = np.linspace(0, total_length, n_points)
    theta_dot_profile = np.zeros(n_points)

    # Fill in theta_dot values for each segment
    current_length = 0

    for seg_type, seg_length in seq:
        # Find indices corresponding to this segment
        start_idx = np.searchsorted(arc_lengths, current_length, side="left")
        end_idx = np.searchsorted(
            arc_lengths, current_length + seg_length, side="right"
        )

        # Set theta_dot values based on segment type
        if seg_type == "L":  # Left turn
            theta_dot_profile[start_idx:end_idx] = 1.0 / rho
        elif seg_type == "R":  # Right turn
            theta_dot_profile[start_idx:end_idx] = -1.0 / rho
        elif seg_type == "S":  # Straight
            theta_dot_profile[start_idx:end_idx] = 0.0

        current_length += seg_length

    return arc_lengths, theta_dot_profile


def generate_path_functions(gamma, xi):

    gamma_d = cs.jacobian(gamma, xi)
    sigma = cs.norm_2(gamma_d)
    e1 = gamma_d / sigma
    # e2 = e1_d / cs.norm_2(e1_d)
    e2 = cs.vertcat(-e1[1], e1[0])
    e1_d = cs.jacobian(e1, xi)
    e2_d = cs.jacobian(e2, xi)

    omega = cs.dot(e1_d, e2)

    sigma_d = cs.jacobian(sigma, xi)
    omega_d = cs.jacobian(omega, xi)

    f_gamma = cs.Function("f_gamma", [xi], [gamma])
    f_e1 = cs.Function("f_e1", [xi], [e1, e1_d], ["xi"], ["e1", "e1_d"])
    f_e2 = cs.Function("f_e2", [xi], [e2, e2_d], ["xi"], ["e2", "e2_d"])
    f_sigma = cs.Function(
        "f_sigma", [xi], [sigma, sigma_d], ["xi"], ["sigma", "sigma_d"]
    )
    f_omega = cs.Function(
        "f_omega", [xi], [omega, omega_d], ["xi"], ["omega", "omega_d"]
    )

    # path = {
    #     "gamma": f_gamma,
    #     "e1": f_e1,
    #     "e2": f_e2,
    #     "sigma": f_sigma,
    #     "omega": f_omega,
    # }

    # evaluate path
    n_eval = 1000
    xi_eval = np.linspace(0, 1, n_eval)
    e1_eval = np.zeros((n_eval, 2))
    e2_eval = np.zeros((n_eval, 2))
    gamma_eval = np.zeros((n_eval, 2))
    sigma_eval = np.zeros((n_eval))
    omega_eval = np.zeros((n_eval))
    xidot_eval = np.zeros(n_eval)
    xi_ref = np.zeros(n_eval)
    eta_ref = np.zeros(n_eval)
    xidot_ref = np.zeros(n_eval)
    for k in range(n_eval):
        gamma_eval[k, :] = np.squeeze(f_gamma(xi_eval[k]))
        e1_eval[k, :] = np.squeeze(f_e1(xi=xi_eval[k])["e1"])
        e2_eval[k, :] = np.squeeze(f_e2(xi=xi_eval[k])["e2"])
        xi_ref[k] = xi_eval[k]  # p2_ref_k[0]
        eta_ref[k] = 0  # p2_ref_k[1]
        sigma_eval[k] = np.squeeze(f_sigma(xi=xi_eval[k])["sigma"])
        omega_eval[k] = np.squeeze(f_omega(xi=xi_eval[k])["omega"])

    path_eval = {
        "xi": xi_eval,
        "gamma": gamma_eval,
        "e1": e1_eval,
        "e2": e2_eval,
        "sigma": sigma_eval,
        "omega": omega_eval,
    }

    # TEST  ------------- -------------
    # interp_type = 'bspline'
    # xi_interp = cs.MX.sym('xi_interp')
    # f_gammax = cs.interpolant('f_gammax', interp_type, [xi_eval], gamma_eval[:, 0])
    # f_gammay = cs.interpolant('f_gammay', interp_type, [xi_eval], gamma_eval[:, 1])
    # f_gamma = cs.Function('f_gamma', [xi_interp], [cs.vertcat(f_gammax(xi_interp), f_gammay(xi_interp))], ["xi"], ["gamma"])

    # f_e1x = cs.interpolant('f_e1x', interp_type, [xi_eval], e1_eval[:, 0])
    # f_e1y = cs.interpolant('f_e1y', interp_type, [xi_eval], e1_eval[:, 1])
    # f_e1 = cs.Function('f_e1', [xi_interp], [cs.vertcat(f_e1x(xi_interp), f_e1y(xi_interp))], ["xi"], ["e1"])

    # f_e2x = cs.interpolant('f_e2x', interp_type, [xi_eval], e2_eval[:, 0])
    # f_e2y = cs.interpolant('f_e2y', interp_type, [xi_eval], e2_eval[:, 1])
    # f_e2 = cs.Function('f_e2', [xi_interp], [cs.vertcat(f_e2x(xi_interp), f_e2y(xi_interp))], ["xi"], ["e2"])

    # f_sigma = cs.interpolant('f_sigma', interp_type, [xi_eval], sigma_eval)
    # f_sigma = cs.Function('f_sigma', [xi_interp], [f_sigma(xi_interp)], ["xi"], ["sigma"])

    # f_omega = cs.interpolant('f_omega', interp_type, [xi_eval], omega_eval)
    # f_omega = cs.Function('f_omega', [xi_interp], [f_omega(xi_interp)], ["xi"], ["omega"])

    path = {
        "gamma": f_gamma,
        "e1": f_e1,
        "e2": f_e2,
        "sigma": f_sigma,
        "omega": f_omega,
    }
    # TEST  ------------- -------------

    return path, path_eval


def get_gamma(gamma_wp, n_upsample, visualize=True):

    # Conver to smooth path
    gamma_wp_upsampled = []
    for i in range(len(gamma_wp) - 1):
        current_wp = gamma_wp[i]
        next_wp = gamma_wp[i + 1]

        for j in range(n_upsample):
            t = j / float(n_upsample)
            interpolated_point = current_wp + t * (next_wp - current_wp)
            gamma_wp_upsampled.append(interpolated_point)

    gamma_wp_upsampled.append(gamma_wp[-1])
    gamma_wp_upsampled = np.array(gamma_wp_upsampled)
    distances = [0]
    for i in range(1, len(gamma_wp_upsampled)):
        dist = np.linalg.norm(gamma_wp_upsampled[i] - gamma_wp_upsampled[i - 1])
        distances.append(distances[-1] + dist)
    distances = np.array(distances)

    # For B-spline, we need to ensure we have enough points and proper setup
    distances = distances / distances[-1]
    interp_type = "bspline"
    interp_x_ca = cs.interpolant(
        "interp_x", interp_type, [distances], gamma_wp_upsampled[:, 0]
    )
    interp_y_ca = cs.interpolant(
        "interp_y", interp_type, [distances], gamma_wp_upsampled[:, 1]
    )

    xi_cs = cs.MX.sym("xi_cs")
    gamma_cs = cs.horzcat(interp_x_ca(xi_cs), interp_y_ca(xi_cs))
    f_path, path_eval = generate_path_functions(gamma=gamma_cs, xi=xi_cs)

    if visualize:
        plt.figure().add_subplot(311)
        plt.plot(path_eval["xi"], path_eval["gamma"][:, 0], label="γ_x")
        plt.plot(path_eval["xi"], path_eval["gamma"][:, 1], label="γ_y")
        plt.legend()
        plt.subplot(312)
        plt.plot(path_eval["xi"], path_eval["sigma"], label="σ")
        plt.legend()
        plt.subplot(313)
        plt.plot(path_eval["xi"], path_eval["omega"], label="ω")
        plt.legend()
        plt.show()

    return f_path, path_eval, gamma_wp


def RK4(x, u, dt, f):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)

    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def dubin_spatial_step(path, d_xi, x, u):
    xi = x[0]
    eta = x[1]
    theta = x[2]
    theta_dot = u

    e1 = path["e1"](xi=xi)["e1"]
    e2 = path["e2"](xi=xi)["e2"]
    sigma = path["sigma"](xi=xi)["sigma"]
    omega = path["omega"](xi=xi)["omega"]

    v = cs.vertcat(cs.cos(theta), cs.sin(theta))
    xi_dot = dubin_spatial_xidot(path=path, x=x)
    eta_dot = cs.dot(e2, v)
    x_next = x + d_xi / xi_dot * cs.vertcat(xi_dot, eta_dot, theta_dot)

    return x_next


def dubin_spatial_xidot(path, x):
    xi = x[0]
    eta = x[1]
    theta = x[2]

    e1 = path["e1"](xi=xi)["e1"]
    sigma = path["sigma"](xi=xi)["sigma"]
    omega = path["omega"](xi=xi)["omega"]

    v = cs.vertcat(cs.cos(theta), cs.sin(theta))
    xi_dot = cs.dot(e1, v) / (sigma - omega * eta)

    return xi_dot


def dubin_spatial_to_cartesian(path, x):
    xi = x[0]
    eta = x[1]
    gamma = path["gamma"](xi)
    e2 = path["e2"](xi=xi)["e2"]
    return np.squeeze(gamma.T + eta * e2)


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":

    case = 1  #! only works for case 1
    solve_nlp = 1
    save = 1

    if case == 1:
        start = (0, 0, pi / 4)  # (x, y, theta) - at origin facing northeast (45°)
        goal = (2, 2, 3 * pi / 4)  # at (2, 2) facing northwest (135°)
        gamma_wp = np.array([[0, 0], [2.1, 0.7], [2, 2]])
        n_upsample = 3
    elif case == 2:
        start = (0, 0, 0)  # (x, y, theta) - at origin facing east (0°)
        goal = (-1, 0, pi + 1e-3)  # 1 unit west, facing west (180°)
        gamma_wp = np.array(
            [
                [0, 0],
                [1.5, -0.25],
                [2.0, 0.5],
                [1.5, 1.5],
                [0.77, 1.7],
                [0.0, 1.5],
                [-0.5, 0.0],
                [-1, 0],
            ]
        )
        n_upsample = 3

    rho = 1.0  # turning radius
    h = 0.02  # step size

    print("=== PATH DATA ===")
    print(f"Start pose: ({start[0]}, {start[1]}, {start[2]*180/pi:.0f}°)")
    print(f"Goal pose:  ({goal[0]}, {goal[1]}, {goal[2]*180/pi:.0f}°)")
    print(f"Turning radius: {rho}")

    # Compute shortest Dubins path
    best_length, best_path = dubins_shortest_path(start, goal, rho)
    arc_lengths, theta_dot_profile = compute_theta_dot_profile(
        best_path, rho=rho, step=h
    )
    path_type, t, p, q = best_path
    waypoints, ths = sample_dubins(start, best_path, rho=rho, step=0.02)
    xs, ys = zip(*waypoints)

    print(f"\nShortest path: {path_type}")
    print(f"Arc 1: {t:.3f} units")
    print(f"Straight: {p:.3f} units")
    print(f"Arc 2: {q:.3f} units")
    print(f"Total length: {best_length:.3f} units")

    # parametric path
    f_path, path_eval, gamma_wp = get_gamma(gamma_wp, n_upsample, visualize=False)

    if solve_nlp:
        # ! ------------------------------- Generate NLP ------------------------------- #

        xi0 = 1e-3
        xif = 1.0 - xi0
        N = 400
        eta_max = 0.25

        x_start = np.array([xi0, 0.0, start[2]])
        x_end = np.array([xif, 0.0, goal[2]])

        xi_grid = np.linspace(xi0, xif, N)
        d_xi = xi_grid[1] - xi_grid[0]

        # decision variables
        nx = 3
        nu = 1
        x = cs.MX.sym("x", N, nx)
        u = cs.MX.sym("u", N - 1, nu)
        x_nlp = cs.vertcat(
            cs.reshape(x, x.size1() * x.size2(), 1),
            cs.reshape(u, u.size1() * u.size2(), 1),
        )  # [px_0,...,px_N,py_0,...py_N,...]
        ff = cs.Function("f_nlp", [x, u], [x_nlp])

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
            if k == N - 1:
                g_nlp = cs.vertcat(g_nlp, x[N - 1, :].T - x_end)
                lbg = cs.vertcat(lbg, [0] * nx)
                ubg = cs.vertcat(ubg, [0] * nx)

            # path parameter constraints
            g_nlp = cs.vertcat(g_nlp, x[k, 0])
            lbg = cs.vertcat(lbg, [xi0])
            ubg = cs.vertcat(ubg, [xif])

            # g_nlp = cs.vertcat(g_nlp, x[k, 1])
            # lbg = cs.vertcat(lbg, [-eta_max])
            # ubg = cs.vertcat(ubg, [eta_max])

            # time law constraints
            xidot = dubin_spatial_xidot(path=f_path, x=x[k, :].T)
            g_nlp = cs.vertcat(g_nlp, xidot)
            lbg = cs.vertcat(lbg, [0.01])
            ubg = cs.vertcat(ubg, [10])

            # input constraints
            if k < N - 1:
                g_nlp = cs.vertcat(g_nlp, u[k, 0])
                lbg = cs.vertcat(lbg, [-1])
                ubg = cs.vertcat(ubg, [1])

            if k < N - 1:
                # continuity
                x_next = dubin_spatial_step(
                    path=f_path, d_xi=d_xi, x=x[k, :].T, u=u[k, :]
                )

                g_nlp = cs.vertcat(g_nlp, (x[k + 1, :].T - x_next))
                lbg = cs.vertcat(lbg, [0] * nx)
                ubg = cs.vertcat(ubg, [0] * nx)

                # cost function
                xidot_next = dubin_spatial_xidot(path=f_path, x=x[k + 1, :])
                dt = 2 * d_xi / (xidot_next + xidot)
                f_nlp += dt  # + 1e-5 * cs.sum1((u[k, :].T) ** 2)  # dt

        # Generate solver
        verbose = True
        nlp_dict = {"x": x_nlp, "f": f_nlp, "g": g_nlp}
        nlp_opts = {
            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            # "ipopt.acceptable_tol": nlp_params["acceptable_tol"],
            "ipopt.max_iter": 100,
            "ipopt.print_level": 5 if verbose else 0,
            "print_time": True,
        }
        nlp_solver = cs.nlpsol("corridor_nlp", "ipopt", nlp_dict, nlp_opts)
        solver = {"solver": nlp_solver, "lbg": lbg, "ubg": ubg}

        # ! --------------------------------- Solve NLP -------------------------------- #
        x_init = np.zeros((N, nx))
        u_init = np.zeros((N - 1, nu))
        for k in range(N):
            e1_k = np.squeeze(f_path["e1"](xi=xi_grid[k])["e1"])
            x_init[k] = np.array([xi_grid[k], 0, np.arctan2(e1_k[1], e1_k[0])])

        xc_init = np.zeros((N, 2))
        for k in range(N):
            xc_init[k] = dubin_spatial_to_cartesian(path=f_path, x=x_init[k])

        x0 = np.hstack(
            [x_init.T.flatten(), u_init.T.flatten()]
        )  # note --> x0 - ff(x_init,u_init)==0
        sol = solver["solver"](
            x0=x0,
            lbg=solver["lbg"],
            ubg=solver["ubg"],
        )
        status = solver["solver"].stats()["success"]
        if not status:
            print("NLP solver failed")
        x_sol = sol["x"]

        # ! --------------------------------- Analysis --------------------------------- #
        # restructure output
        x = np.squeeze(x_sol)[: nx * N].reshape(nx, N).T
        u = np.squeeze(x_sol)[nx * N :].reshape(nu, N - 1).T

        # convert from spatial to cartesian coordinates and get other variables
        p = np.zeros((N, 2))
        v = np.zeros((N, 2))
        theta = np.zeros(N)
        theta_dot = u.copy()
        xi = np.zeros(N)
        eta = np.zeros(N)
        xidot = np.zeros(N)
        for k in range(N):
            p[k] = dubin_spatial_to_cartesian(path=f_path, x=x[k])
            v[k] = np.array([np.cos(x[k, 2]), np.sin(x[k, 2])])
            theta[k] = x[k, 2]
            xi[k] = x[k, 0]
            eta[k] = x[k, 1]
            xidot[k] = dubin_spatial_xidot(path=f_path, x=x[k])

    # Visualizet trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    arrow_scale = 0.4
    plt.arrow(
        start[0],
        start[1],
        arrow_scale * np.cos(start[2]),
        arrow_scale * np.sin(start[2]),
        head_width=0.15,
        head_length=0.15,
        fc="green",
        ec="green",
        linewidth=2,
    )
    plt.arrow(
        goal[0],
        goal[1],
        arrow_scale * np.cos(goal[2]),
        arrow_scale * np.sin(goal[2]),
        head_width=0.15,
        head_length=0.15,
        fc="red",
        ec="red",
        linewidth=2,
    )

    circle1 = plt.Circle(
        (start[0] - rho * np.sin(start[2]), start[1] + rho * np.cos(start[2])),
        rho,
        fill=False,
        color="green",
        alpha=0.3,
        linestyle="--",
        label="Start turning circle",
    )
    circle2 = plt.Circle(
        (goal[0] - rho * np.sin(goal[2]), goal[1] + rho * np.cos(goal[2])),
        rho,
        fill=False,
        color="red",
        alpha=0.3,
        linestyle="--",
        label="Goal turning circle",
    )
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    ax.scatter([start[0]], [start[1]], c="green", s=200, label="Start", marker="o")
    ax.scatter([goal[0]], [goal[1]], c="red", s=200, label="Goal", marker="s")
    # ax.plot(gamma_wp[:, 0], gamma_wp[:, 1], color='black', alpha=0.25, linestyle='--', marker='o', label='Linear γ')
    # ax.plot(gamma[:, 0], gamma[:, 1], color='m', alpha=0.5, linestyle='-', label='Smooth γ')
    plot_frames(
        ax=ax,
        r=path_eval["gamma"],
        e1=path_eval["e1"],
        e2=path_eval["e2"],
        e3=np.zeros_like(path_eval["e1"]),
        scale=0.1,
        interval=0.98,
        ax_equal=False,
        planar=True,
    )
    ax.plot(xs, ys, "b-", linewidth=3, label=f"Dubins Path")
    ax.plot(gamma_wp[:, 0], gamma_wp[:, 1], "k--", alpha=0.5, label=f"Dubins Path")
    if solve_nlp:
        ax.plot(xc_init[:, 0], xc_init[:, 1], "k-", alpha=0.4, label=f"Initialization")
        ax.plot(p[:, 0], p[:, 1], "m-", label=f"Path-Parametric")
        ax.set_aspect("equal")
        ax.legend()

        # Visualize state and input variables
        fig = plt.figure()
        ax = fig.add_subplot(511)
        ax.plot(theta, label=r"$\theta$")
        ax.legend()

        ax = fig.add_subplot(512)
        ax.plot(theta_dot, label=r"$\dot{\theta}$")
        ax.legend()

        ax = fig.add_subplot(513)
        ax.plot(xi, label=r"$\xi$")
        ax.legend()

        ax = fig.add_subplot(514)
        ax.plot(eta, label=r"$\eta$")
        ax.plot(eta_max * np.ones(N), "k--")
        ax.plot(-eta_max * np.ones(N), "k--")
        ax.legend()

        ax = fig.add_subplot(515)
        ax.plot(xidot, label=r"$\dot{\xi}$")
        ax.legend()

    plt.show()

    if save:
        save_pickle(
            path="/Users/jonarrizabalaga/PathParam/examples/dubins/paper/data",
            file_name="dubins",
            data={
                "x": x,
                "u": u,
                "xi": xi,
                "eta": eta,
                "theta": theta,
                "theta_dot": theta_dot,
                "xidot": xidot,
                "p": p,
                "rho": rho,
                "start": start,
                "goal": goal,
                "gamma_wp": gamma_wp,
                "eta_max": eta_max,
                "path_eval": path_eval,
                "p_dubins": np.vstack((xs, ys)),
            },
        )
