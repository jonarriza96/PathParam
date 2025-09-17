# %%
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from scipy.optimize import fsolve

from examples.robotic_manipulator.src.utils import load_pickle, get_package_path


def solve_brachistochrone(p_start, p_end, n_points=100, g=9.81):
    """
    Solve the brachistochrone problem to find the curve of fastest descent.

    The brachistochrone curve is a cycloid that minimizes the time of descent
    under gravity between two points.

    Parameters:
    -----------
    p_start : array-like
        Starting point [x0, y0]
    p_end : array-like
        Ending point [x1, y1]
    n_points : int
        Number of points to generate along the curve
    g : float
        Gravitational acceleration (default: 9.81 m/s²)

    Returns:
    --------
    x_curve : ndarray
        X coordinates of the brachistochrone curve
    y_curve : ndarray
        Y coordinates of the brachistochrone curve
    time_optimal : float
        Optimal descent time
    """
    x0, y0 = p_start
    x1, y1 = p_end

    # The brachistochrone is a cycloid. We need to find the parameters
    # of the cycloid that passes through both points.

    # For a cycloid: x = R*(t - sin(t)), y = -R*(1 - cos(t))
    # where R is the radius and t is the parameter

    def cycloid_equations(params):
        R, t1 = params
        # Point 1 should be at t=0 (start)
        x_start_calc = 0  # R*(0 - sin(0)) = 0
        y_start_calc = 0  # -R*(1 - cos(0)) = 0

        # Point 2 should be at t=t1 (end)
        x_end_calc = R * (t1 - np.sin(t1))
        y_end_calc = -R * (1 - np.cos(t1))

        # Adjust for actual start position
        x_end_calc += x0
        y_end_calc += y0

        return [x_end_calc - x1, y_end_calc - y1]

    # Initial guess for R and t1
    dx = x1 - x0
    dy = y1 - y0
    R_guess = abs(dy) / 2  # rough estimate
    t1_guess = np.pi  # half cycle

    # Solve for the cycloid parameters
    R, t1 = fsolve(cycloid_equations, [R_guess, t1_guess])

    # Generate the curve points
    t = np.linspace(0, t1, n_points)
    x_curve = x0 + R * (t - np.sin(t))
    y_curve = y0 - R * (1 - np.cos(t))

    # Calculate the optimal time
    # For a cycloid, the time is T = sqrt(R/g) * t1
    time_optimal = np.sqrt(R / g) * t1

    return x_curve, y_curve, time_optimal


# def solve_brachistochrone_pathparametric(e1):


if __name__ == "__main__":
    # -------------------------------- Parameters -------------------------------- #
    p_start = np.array([0, 0])
    p_end = np.array([1, -1])
    p_end = p_end / np.linalg.norm(p_end)
    n_eval = 100

    # ----------------------------------- Solve ---------------------------------- #
    gamma = lambda xi: p_start + (p_end - p_start) * xi
    gamma_eval = np.array([gamma(xi) for xi in np.linspace(0, 1, n_eval)])

    ##### Solve the brachistochrone problem #####
    x_curve, y_curve, optimal_time = solve_brachistochrone(p_start, p_end)

    # ##### Solve brachistochrone problem pathparametric #####
    # xi_pp = np.linspace(0, 1, n_eval)

    # # 1 calculate velocity in closed form
    # e1 = (p_end - p_start) / np.linalg.norm(p_end - p_start)
    # e2 = np.array([-e1[1], e1[0]])
    # R = np.vstack((e1, e2))
    # g = np.array([0, -9.81])

    # # a = np.dot(g.T, e1)\
    # # b = np.dot(g.T, e2)
    # # v_pp = np.sqrt(2*a*xi_pp)*(e1+b/a*e2)[:,None]
    # # v_pp = np.sqrt(2*(np.dot(e1.T,g)*xi_pp))*e1[:,None]
    # # v_pp = R @ v_pp
    # # delta_y = np.abs(p_end[1] - p_start[1])
    # # v_pp = (np.sqrt(2*9.81*delta_y*xi_pp)[:,None]*e1[None,:]).T
    # theta = np.pi/4
    # L= 1
    # g = 9.81
    # b = -L
    # n = -L*g*np.cos(theta)
    # v_pp = b*e1 + n*e2

    # # 2 obtain position from velocity
    # p_pp = np.zeros((2, n_eval))
    # p_pp[:, 0] = p_start
    # for i in range(n_eval - 1):
    #     p_pp[:, i + 1] = p_pp[:, i] + v_pp * (xi_pp[i + 1] - xi_pp[i])

    # %%
    # --------------------------------- Visualize -------------------------------- #
    # Plot the results
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    # Plot the optimal curve
    ax.plot(
        x_curve,
        y_curve,
        "b-",
        linewidth=2,
        label=f"Brachistochrone (T={optimal_time:.3f}s)",
    )
    # ax.plot(p_pp[0, :], p_pp[1, :], 'm-', linewidth=2, label=f'Path parametric')

    # Plot start and end points
    ax.plot([p_start[0]], [p_start[1]], "go", markersize=8, label="Start")
    ax.plot([p_end[0]], [p_end[1]], "ro", markersize=8, label="End")

    # Plot straight line for comparison
    # ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', alpha=0.7, label='Straight line')
    ax.plot(gamma_eval[:, 0], gamma_eval[:, 1], "k--", alpha=0.7, label="γ")

    ax.grid(True, alpha=0.3)
    ax.legend()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Brachistochrone Problem Solution')
    ax.axis("off")
    ax.set_aspect("equal")
    # plt.show()

    plt.savefig(
        get_package_path()
        + "/examples/brachistochrone/paper/brachistochrone_trajectory.pdf",
        dpi=300,
    )

    print(f"Optimal descent time: {optimal_time:.3f} seconds")
