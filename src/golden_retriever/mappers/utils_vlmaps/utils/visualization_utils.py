import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def plot_3d_point_could(point_cloud, pc_color):
    fig = plt.figure()
    ax = Axes3D(fig)

    # creating the plot
    if isinstance(pc_color, str):
        ax.scatter(
            point_cloud[:, :, 0],
            point_cloud[:, :, 1],
            point_cloud[:, :, 2],
            color=pc_color,
        )
    elif isinstance(pc_color, np.ndarray):
        # Flat the x, y, z and add the color (RGB)
        x = point_cloud[:, :, 0].reshape(-1, 1)
        y = point_cloud[:, :, 1].reshape(-1, 1)
        z = point_cloud[:, :, 2].reshape(-1, 1)
        color = pc_color.reshape(-1, 3)
        # Colored point cloud
        color_point_cloud = np.concatenate([x, y, z, color], axis=1)
        # Plot
        ax.scatter(
            color_point_cloud[:, 0],
            color_point_cloud[:, 1],
            color_point_cloud[:, 2],
            c=color_point_cloud[:, 3:] / 255.0,
            s=1,
        )

    # setting title and labels
    ax.set_title("3D plot")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    # displaying the plot
    plt.show()


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)

    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def get_mask_palette(
    npimg, new_palette, out_label_flag=False, labels=None, ignore_ids_list=()
):
    """Get image color palette for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    patches = []
    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mp.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches
