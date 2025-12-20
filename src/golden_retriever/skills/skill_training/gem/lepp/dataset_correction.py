import os

import lepp.transformations as transformations
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lepp.clip_preprocess import CLIP_processor
from lepp.parser import parse_instruction


def rotatePixelCoordinate(image_shape: tuple, pixel_xy: np.array, rotate_angle: float):
    """
    We define x, y to be row and column respectively
    rotate_angle is in rad
    """
    image_shape = np.array(image_shape)
    image_center = image_shape[:2] // 2
    rotation_mat = np.array(
        [
            [np.cos(rotate_angle), -np.sin(rotate_angle)],
            [np.sin(rotate_angle), np.cos(rotate_angle)],
        ]
    )
    pixel_xy = (pixel_xy - image_center).reshape(2, 1)
    length = np.sqrt(pixel_xy[0] ** 2 + pixel_xy[1] ** 2)
    result = rotation_mat.dot(pixel_xy).reshape(
        2,
    ) + np.array([image_center[1], image_center[0]])
    result_x = np.clip(result[0], 0, image_shape[1])
    result_y = np.clip(result[1], 0, image_shape[0])
    return np.array([result_x, result_y]).astype(int)


def rotateImage90(image: np.array):
    return np.rot90(image)


def visualize(
    rgb,
    depth,
    p0,
    p1,
    p0_theta,
    p1_theta,
    clip_features_text,
    clip_features_image,
    clip_features,
):
    fig, ax = plt.subplots(2, 5)
    ax[0][0].imshow(rgb.astype(int))
    ax[0][0].set_title("RGB")
    ax[0][1].imshow(depth)
    ax[0][1].set_title("depth")
    ax[0][2].imshow(
        clip_features[
            ...,
            1:2,
        ]
    )
    ax[0][2].set_title("combined clip_features")
    ax[0][3].imshow(clip_features_text)
    ax[0][3].set_title("text clip_features")
    ax[0][4].imshow(clip_features_image)
    ax[0][4].set_title("image clip_features")

    ax[1][2].imshow(
        (
            clip_features[
                ...,
                1:2,
            ]
            * rgb
        ).astype(int)
    )
    ax[1][2].set_title("combined clip_features")
    ax[1][3].imshow((clip_features_text * rgb).astype(int))
    ax[1][3].set_title("text clip_features")
    ax[1][4].imshow((clip_features_image * rgb).astype(int))
    ax[1][4].set_title("image clip_features")
    # ax[2].imshow(clip_feature_pick[..., 0])
    # ax[3].imshow(clip_feature_place[..., 0])
    p0_theta = (p0_theta + 2 * np.pi) % (2 * np.pi)
    p1_theta = p0_theta + p1_theta
    # print('row, column, rotz:', p0[0], p0[1])
    ax[0][0].plot(p0[1], p0[0], marker="o", color="green")
    ax[0][0].plot(p1[1], p1[0], marker="x", color="red")
    arrow_length = 30
    ax[0][0].arrow(
        p0[1],
        p0[0],
        arrow_length * np.cos(p0_theta),
        -arrow_length * np.sin(p0_theta),
        width=0.005,
        color="green",
    )
    ax[0][0].arrow(
        p1[1],
        p1[0],
        arrow_length * np.cos(p1_theta),
        -arrow_length * np.sin(p1_theta),
        width=0.005,
        color="red",
    )
    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(1)


def get_crop(rgb, pixel_xy, clip_kernel_size):
    pad_size = clip_kernel_size // 2
    pad_rgb = (
        F.pad(
            input=torch.from_numpy(rgb).permute(2, 0, 1),
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode="replicate",
        )
        .permute(1, 2, 0)
        .numpy()
    )

    x, y = np.array(pixel_xy) + pad_size
    return pad_rgb[x - pad_size : x + pad_size, y - pad_size : y + pad_size, :]


if __name__ == "__main__":
    final_name = "pick-part-in-box-real-old.npy"
    file_list = [
        "pick-part-in-box-real-old.npy",
    ]
    datapath = "./data"
    task_name = "pick-part-in-box-real"
    processor = CLIP_processor()
    dataout = []
    raw_data_convertion = False
    for file in file_list:
        data = np.load(os.path.join(datapath, file), allow_pickle=True).tolist()
        print(file)
        for i, episode in enumerate(data):
            print(f"{i}/{len(data)}")
            if file == "pick-part-in-box-real-demo4.npy":
                episode = [episode]
                data[i] = episode
            # print(episode[0]['instruction'])
            pick_ins, place_ins = parse_instruction(
                task_name, episode[0]["instruction"]
            )

            # episode[0]['p0_theta'], episode[0]['p1_theta'] = -episode[0]['p0_theta'], -episode[0]['p1_theta']
            episode[0]["p0_theta"], episode[0]["p1_theta"] = (
                transformations.euler_from_quaternion(episode[0]["quat0"], axes="szyx")[
                    0
                ],
                transformations.euler_from_quaternion(episode[0]["quat1"], axes="szyx")[
                    0
                ],
            )
            clip_pp = processor.get_clip_feature(
                episode[0]["rgb"], episode[0]["instruction"]
            )[0]

            clip_kernel_size = 40
            # pad_rgb = F.pad(input=torch.from_numpy(episode[0]['rgb']).permute(2,0,1), pad=(pad_size, pad_size, pad_size, pad_size), mode='replicate').permute(1,2,0).numpy()
            pick_crop = get_crop(episode[0]["rgb"], episode[0]["p0"], clip_kernel_size)
            # pick_crop = episode[0]['rgb'][episode[0]['p0'][0]-pad_size:episode[0]['p0'][0]+pad_size, episode[0]['p0'][1]-pad_size:episode[0]['p0'][1]+pad_size, :]
            (
                clip_pick_text,
                clip_pick_image,
                clip_pick,
                pick_text_emb,
            ) = processor.get_clip_feature_from_text_and_image(
                episode[0]["rgb"], pick_ins, pick_crop
            )
            # place_crop = episode[0]['rgb'][episode[0]['p1'][0]-pad_size:episode[0]['p1'][0]+pad_size, episode[0]['p1'][1]-pad_size:episode[0]['p1'][1]+pad_size, :]
            place_crop = get_crop(episode[0]["rgb"], episode[0]["p1"], clip_kernel_size)
            (
                clip_place_text,
                clip_place_image,
                clip_place,
                place_text_emb,
            ) = processor.get_clip_feature_from_text_and_image(
                episode[0]["rgb"], place_ins, place_crop
            )
            clip_features = np.concatenate([clip_pp, clip_pick, clip_place], axis=2)
            episode[0]["clip_features"] = clip_features
            visualize(
                episode[0]["rgb"],
                episode[0]["depth"],
                episode[0]["p0"],
                episode[0]["p1"],
                episode[0]["p0_theta"],
                episode[0]["p1_theta"],
                clip_pick_text,
                clip_pick_image,
                clip_features,
            )
            # print(1)
            if raw_data_convertion:
                image_shape_old = episode[0]["rgb"].shape
                episode[0]["rgb"] = rotateImage90(episode[0]["rgb"])
                episode[0]["depth"] = rotateImage90(
                    np.clip(episode[0]["depth"], 0, 0.2)
                )
                episode[0]["p1"] = rotatePixelCoordinate(
                    image_shape_old, episode[0]["p1"], np.pi / 2
                )
                episode[0]["p1_theta"] = episode[0]["p1_theta"] - episode[0]["p0_theta"]
                episode[0]["p0"] = rotatePixelCoordinate(
                    image_shape_old, episode[0]["p0"], np.pi / 2
                )
                episode[0]["p0_theta"] = episode[0]["p0_theta"] + np.pi / 2
                episode[0]["clip_features"] = rotateImage90(clip_features)
            episode[0]["pick_crop"] = pick_crop
            episode[0]["place_crop"] = place_crop
            episode[0]["pick_text_emb"] = pick_text_emb
            episode[0]["place_text_emb"] = place_text_emb
            # visualize(episode[0]['rgb'], episode[0]['depth'], episode[0]['p0'], episode[0]['p1'], episode[0]['p0_theta'], episode[0]['p1_theta'], episode[0]['clip_features'])
            # print(1)

        dataout = dataout + data

    np.save(os.path.join(datapath, final_name), dataout)
