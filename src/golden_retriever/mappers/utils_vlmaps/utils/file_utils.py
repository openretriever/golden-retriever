import os

import numpy as np


def create_offline_dataset_dirs(dir_path: str) -> None:
    # Create saving directories
    if not os.path.exists(dir_path):
        os.makedirs(f"{dir_path}/RGB")
        os.makedirs(f"{dir_path}/Depth")
        os.makedirs(f"{dir_path}/Pose")


def load_offline_batch_episodes(data_path: str, batch_size: int):
    total_episode_num = len(os.listdir(data_path))
    batch_episode_indices = np.random.choice(range(total_episode_num), batch_size)

    episodes = []
    max_len = 0
    for idx in batch_episode_indices:
        episode = np.load(f"{data_path}/episode_{idx}.npz")
        episodes.append(
            {
                "rgb": episode["rgb"],
                "depth": episode["depth"],
                "pose": episode["abs_pose"],
                "traj_len": episode["rgb"].shape[0],
            }
        )

        max_len = (
            episodes[-1]["traj_len"] if episodes[-1]["traj_len"] > max_len else max_len
        )

    # pad each episode
    for idx, episode in enumerate(episodes):
        pad_num = max_len - episode["traj_len"]

        if pad_num:
            episodes[idx]["rgb"] = np.pad(
                episodes[idx]["rgb"],
                pad_width=((0, pad_num), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )
            episodes[idx]["depth"] = np.pad(
                episodes[idx]["depth"],
                pad_width=((0, pad_num), (0, 0), (0, 0)),
                mode="edge",
            )
            episodes[idx]["pose"] = np.pad(
                episodes[idx]["pose"], pad_width=((0, pad_num), (0, 0)), mode="edge"
            )

    # construct the batch data
    batch_episodes = {"rgb": [], "depth": [], "pose": [], "pad_traj_len": max_len}
    for episode in episodes:
        batch_episodes["rgb"].append(episode["rgb"])
        batch_episodes["depth"].append(episode["depth"])
        batch_episodes["pose"].append(episode["pose"])

    batch_episodes["rgb"] = np.array(batch_episodes["rgb"])
    batch_episodes["depth"] = np.array(batch_episodes["depth"])
    batch_episodes["pose"] = np.array(batch_episodes["pose"])

    return batch_episodes


def create_directories(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_files(dir_path: str, sort_ord: str) -> list:
    files = os.listdir(dir_path)

    if sort_ord == "descending":
        files = sorted(
            files, key=lambda x: int(x.split(".")[0].split("_")[1]), reverse=True
        )
    elif sort_ord == "ascending":
        files = sorted(
            files, key=lambda x: int(x.split(".")[0].split("_")[1]), reverse=False
        )
    else:
        pass
    return files


def load_offline_data(data_path: str, idx: int) -> list:
    """Load the offline data from the dataset"""
    _rgb = np.load(f"{data_path}/RGB/rgb_img_{idx}.npy", allow_pickle=True)
    _depth = np.load(f"{data_path}/Depth/depth_img_{idx}.npy", allow_pickle=True)
    _pose = np.load(f"{data_path}/Pose/pose_{idx}.npy", allow_pickle=True)
    return [_rgb, _depth, _pose]


def load_offline_episode_data(data_path: str):
    """Load the offline data from the dataset"""
    rgb_list, depth_list, pose_list = [], [], []

    data_sample_num = len(os.listdir(f"{data_path}/RGB"))

    for idx in range(data_sample_num):
        rgb_list.append(
            np.load(f"{data_path}/RGB/rgb_img_{idx}.npy", allow_pickle=True)
        )
        depth_list.append(
            np.load(f"{data_path}/Depth/depth_img_{idx}.npy", allow_pickle=True)
        )
        pose_list.append(np.load(f"{data_path}/Pose/pose_{idx}.npy", allow_pickle=True))

    return (
        np.array(rgb_list),
        np.array(depth_list),
        np.array(pose_list),
        data_sample_num,
    )
