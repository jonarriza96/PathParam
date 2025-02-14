import numpy as np
import matplotlib.pyplot as plt


def axis_equal(X, Y, Z, ax=None):
    """
    Sets axis bounds to "equal" according to the limits of X,Y,Z.
    If axes are not given, it generates and labels a 3D figure.

    Args:
        X: Vector of points in coord. x
        Y: Vector of points in coord. y
        Z: Vector of points in coord. z
        ax: Axes to be modified

    Returns:
        ax: Axes with "equal" aspect


    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - 1.2 * max_range, mid_x + 1.2 * max_range)
    ax.set_ylim(mid_y - 1.2 * max_range, mid_y + 1.2 * max_range)
    ax.set_zlim(mid_z - 1.2 * max_range, mid_z + 1.2 * max_range)

    return ax


def plot_frames(
    r, e1, e2, e3, interval=0.9, scale=1.0, ax=None, ax_equal=True, planar=False
):
    """
    Plots the moving frame [e1,e2,e3] of the curve r. The amount of frames to
    be plotted can be controlled with "interval".

    Args:
        r: Vector of 3d points (x,y,z) of curve
        e1: Vector of first component of frame
        e2: Vector of second component of frame
        e3: Vector of third component of frame
        interval: Percentage of frames to be plotted, i.e, 1 plots a frame in
                  every point of r, while 0 does not plot any.
        scale: Float to size components of frame
        ax: Axis where plot will be modified

    Returns:
        ax: Modified plot
    """
    # scale = 0.1
    nn = r.shape[0]
    tend = r + e1 * scale
    nend = r + e2 * scale
    bend = r + e3 * scale
    # interval = 1
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")

    if planar:
        if interval == 1:
            rng = range(nn)
        else:
            rng = range(0, nn, int(nn * (1 - interval)))
        for i in rng:  # if nn >1 else 1):
            ax.plot([r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], "r")
            ax.plot([r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], "g")

            # ax.plot([r[i, 0], tend[i, 0]], [r[i, 2], tend[i, 2]], "r")  # , linewidth=2)
            # ax.plot([r[i, 0], bend[i, 0]], [r[i, 2], bend[i, 2]], "g")  # , linewidth=2)
        ax.set_aspect("equal")

    else:
        if ax_equal:
            ax = axis_equal(r[:, 0], r[:, 1], r[:, 2], ax=ax)
        if interval == 1:
            rng = range(nn)
        else:
            rng = range(0, nn, int(nn * (1 - interval)) if nn > 1 else 1)

        for i in rng:
            ax.plot(
                [r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], [r[i, 2], tend[i, 2]], "r"
            )
            ax.plot(
                [r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], [r[i, 2], nend[i, 2]], "g"
            )
            ax.plot(
                [r[i, 0], bend[i, 0]], [r[i, 1], bend[i, 1]], [r[i, 2], bend[i, 2]], "b"
            )

    return ax


def visualize_moving_frame(p, R, ax=None, title=None, planar=False):
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")
    if planar:
        plot_frames(
            ax=ax,
            r=p,
            e1=R[:, :, 0],
            e2=R[:, :, 1],
            e3=R[:, :, 2],
            scale=0.1,
            interval=0.98,
            ax_equal=False,
            planar=True,
        )
        ax.set_aspect("equal")
    else:
        plot_frames(
            ax=ax,
            r=p,
            e1=R[:, :, 0],
            e2=R[:, :, 1],
            e3=R[:, :, 2],
            scale=0.1,
            interval=0.98,
            ax_equal=False,
        )
        axis_equal(p[:, 0], p[:, 1], p[:, 2], ax=ax)
    if title is not None:
        ax.set_title(title)

    return ax


def visualize_angular_velocity(
    xi, omega, alpha, j, ax=None, xlabel=None, ylabel=None, title=None
):

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots(2, 1)

    ax[0].plot(xi, omega)
    ax[1].plot(xi, j)

    if xlabel is not None:
        ax[1].set_xlabel(xlabel[0])

    if ylabel is not None:
        ax[0].set_ylabel(r"$\omega^{\Gamma}$")
        ax[1].set_ylabel(r"$j^{\Gamma}$")

    if title is not None:
        ax[0].set_title(title)

    return ax


def visualize(xi, R, Rd, Rdd, omega, alpha, j, title=None):

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["r", "g", "b"])

    # moving frame data
    plt.figure()
    plt.subplot(3, 4, 1)
    plt.plot(xi, R[:, :, 0])
    plt.ylabel(r"$e_1$")
    plt.subplot(3, 4, 2)
    plt.plot(xi, R[:, :, 1])
    plt.ylabel(r"$e_2$")
    plt.subplot(3, 4, 3)
    plt.plot(xi, R[:, :, 2])
    plt.ylabel(r"$e_3$")

    plt.subplot(3, 4, 5)
    plt.plot(xi, Rd[:, :, 0])
    plt.ylabel(r"$e_1^{'}$")
    plt.subplot(3, 4, 6)
    plt.plot(xi, Rd[:, :, 1])
    plt.ylabel(r"$e_2^{'}$")
    plt.subplot(3, 4, 7)
    plt.plot(xi, Rd[:, :, 2])
    plt.ylabel(r"$e_3^{'}$")

    plt.subplot(3, 4, 9)
    plt.plot(xi, Rdd[:, :, 0])
    plt.ylabel(r"$e_1^{''}$")
    plt.subplot(3, 4, 10)
    plt.plot(xi, Rdd[:, :, 1])
    plt.ylabel(r"$e_2^{''}$")
    plt.subplot(3, 4, 11)
    plt.plot(xi, Rdd[:, :, 2])
    plt.ylabel(r"$e_3^{''}$")

    plt.subplot(344)
    plt.plot(xi, omega)
    plt.ylabel(r"$\omega^{\Gamma}$")
    plt.subplot(348)
    plt.plot(xi, alpha)
    plt.ylabel(r"$\alpha^{\Gamma}$")
    plt.subplot(3, 4, 12)
    plt.plot(xi, j)
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$j^{\Gamma}$")

    if title is not None:
        plt.suptitle(title)
