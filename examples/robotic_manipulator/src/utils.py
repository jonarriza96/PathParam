import numpy as np
import matplotlib.pyplot as plt

import os
import pickle


def get_package_path():
    # package_path = subprocess.run(
    #     "echo $UPPC_2JOINT_PATH", shell=True, capture_output=True, text=True
    # ).stdout.strip("\n")
    package_path = os.getenv("PATHPARAM_PATH")
    return package_path


def save_pickle(path, file_name, data):
    pickle_path = path + "/" + file_name + ".pickle"
    with open(pickle_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data


def get_colors():
    color_g = (0 / 255, 204 / 255, 0 / 255)
    color_p = (204 / 255, 51 / 255, 204 / 255)

    return color_g, color_p


def plot_two_link_manipulator(ax, joint1, joint2, color="g"):

    base = (0, 0)  # Base of the manipulator
    color_g, color_p = get_colors()
    if color == "g":
        color = color_g
    elif color == "p":
        color = color_p

    # Plot the first link from the base to the first joint
    ax.plot(
        [base[0], joint1[0]],
        [base[1], joint1[1]],
        "-",
        color="gray",
        linewidth=2,
        label="Link 1",
    )
    # Plot the second link from the first joint to the second joint
    ax.plot(
        [joint1[0], joint2[0]],
        [joint1[1], joint2[1]],
        "-",
        # linewidth=2,
        color="gray",
        label="Link 2",
    )

    # Plot the joints as circles
    ax.plot(base[0], base[1], "ko", markersize=10, label="Base", alpha=0.5)
    ax.plot(
        joint1[0], joint1[1], "o", color="gray", markersize=8, label="Joints", alpha=0.5
    )
    ax.plot(joint2[0], joint2[1], "o", color=color, markersize=8, alpha=0.5)

    # Ground
    ax.plot([-0.1, 0.1], [-0.025, -0.025], "k-")  # , linewidth=2)
    for i in np.linspace(-0.15, 0.05, num=10):
        ax.plot(
            [i, i + 0.05], [-0.06, -0.03], "k-"
        )  # , lw=1)  # Hatching lines for ground

    return ax
