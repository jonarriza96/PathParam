# %%
from examples.robotic_manipulator.src.utils import (
    get_package_path,
    load_pickle,
    plot_two_link_manipulator,
    get_colors,
)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg


def visualize(X_t, X_ref_t, X_f, X_ref_f, t, xi):
    fig = plt.clf()
    # ax = plt.gca()

    plt.subplot(121)
    ax = plt.gca()
    ax.plot(X_ref_t[-1, 2], X_ref_t[-1, 3], "ro", alpha=1.0)
    ax = plot_two_link_manipulator(
        ax, joint1=X_t[-1, 0:2], joint2=X_t[-1, 2:4], color="m"
    )
    ax.plot(X_t[:, 2], X_t[:, 3], "-", color=color_p, alpha=0.75, lw=1.5)
    ax.plot(path[:, 0], path[:, 1], "k--", alpha=1.0, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_axis_off()
    ax.set_ylim([-0.53, 1.6])
    plt.title(f"Tracking\nt = {t:.2f} s", color=color_p)

    if 3.0 <= t and t <= 6.0:
        # text
        plt.text(
            -0.3,
            1.45,
            "Disturbance",
            fontsize=8,
        )

        # image
        img = mpimg.imread(
            get_package_path()
            + "/examples/robotic_manipulator/paper/example1/images/lightning.png"
        )
        scale_factor = 0.03
        img_resized = img[:: int(1 / scale_factor), :: int(1 / scale_factor)]

        x_pixel, y_pixel = ax.transData.transform((-0.45, 1.37))

        plt.figimage(
            img_resized,
            xo=x_pixel,
            yo=y_pixel,
        )

    # ax = fig.add_subplot(121)
    plt.subplot(122)
    ax = plt.gca()
    ax.plot(X_ref_f[-1, 2], X_ref_f[-1, 3], "ro", alpha=1.0)
    ax = plot_two_link_manipulator(
        ax, joint1=X_f[-1, 0:2], joint2=X_f[-1, 2:4], color="g"
    )
    ax.plot(X_f[:, 2], X_f[:, 3], "-", color=color_g, alpha=0.75, lw=1.5)
    ax.plot(path[:, 0], path[:, 1], "k--", alpha=1.0, linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_axis_off()
    ax.set_ylim([-0.53, 1.6])
    plt.title("Following\n" + r"$\xi$=" + f"{xi:.2f}", color=color_g)
    if 3.0 <= t and t <= 6.0:
        # text
        plt.text(
            -0.25,
            1.45,
            "Disturbance",
            fontsize=8,
        )

        # image
        img = mpimg.imread(
            get_package_path()
            + "/examples/robotic_manipulator/paper/example1/images/lightning.png"
        )
        scale_factor = 0.03
        img_resized = img[:: int(1 / scale_factor), :: int(1 / scale_factor)]

        x_pixel, y_pixel = ax.transData.transform((-0.4, 1.37))

        plt.figimage(
            img_resized,
            xo=x_pixel,
            yo=y_pixel,
        )

    # plt.suptitle(f"t = {t:.2f} s")


color_g, color_p = get_colors()

# case = "tracking_nodisturbance"
# case = "following_nodisturbance"
case_t = "tracking_disturbance"
case_f = "following_disturbance"

# Load data
data_t = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example1/data/"
    + case_t
    + ".pickle"
)
X_t = data_t["X"]
U_t = data_t["U"]
X_ref_t = data_t["X_ref"]
t_sim_t = data_t["t_sim"]
path = data_t["path"]

data_f = load_pickle(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example1/data/"
    + case_f
    + ".pickle"
)
X_f = data_f["X"]
U_f = data_f["U"]
X_ref_f = data_f["X_ref"]
xi_f = data_f["xi_ref"]
t_sim_f = data_f["t_sim"]
path_f = data_f["path"]


# i = -2
# visualize(
#     X_t=X_t[:i],
#     X_ref_t=X_ref_t[:i],
#     X_f=X_f[:i],
#     X_ref_f=X_ref_f[:i],
#     title=f"t = {t_sim_t[i]:.2f} s",
# )


# visualize
print("Saving animations ...")
fig = plt.figure()
animation = FuncAnimation(
    fig,
    lambda i: visualize(
        X_t=X_t[:i],
        X_ref_t=X_ref_t[:i],
        X_f=X_f[:i],
        X_ref_f=X_ref_f[:i],
        t=t_sim_t[i],
        xi=xi_f[i],
    ),
    frames=range(1, len(X_t) - 1),
)
animation.save(
    get_package_path()
    + "/examples/robotic_manipulator/paper/example1/images/"
    + "disturbance"
    + ".gif",
    writer="ffmpeg",
    fps=len(X_t) // 4,
)
print("Done")
