import time

import IPython.terminal.debugger as debug
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils_vlmaps.utils.depth_utils as du
from torch import Tensor
from utils_vlmaps.mapping import VisLangMapper
from utils_vlmaps.segmentation import Segmentation
from utils_vlmaps.utils.file_utils import load_offline_episode_data
from utils_vlmaps.utils.parser import YamlParser
from utils_vlmaps.utils.pose_utils import get_rot_and_trans_matrices
from utils_vlmaps.utils.visualization_utils import get_mask_palette, get_palette


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to tensor and move to GPU"""
    return torch.from_numpy(arr).float().to(device)


def main(cfg):
    def get_semantic_mask_and_patch(labels):
        """Get mask for visualization"""
        sem_patches = []
        sem_maps = []
        for label in labels:
            palette = get_palette(len(vlm.lang_labels))
            palette[0:3] = [255, 0, 0]
            mask, patches = get_mask_palette(
                label, palette, out_label_flag=True, labels=vlm.lang_labels
            )

            sem_patches.append(patches)
            sem_maps.append(np.array(mask.convert("RGBA"))[..., 0:3])
        return sem_maps, sem_patches

    def visualize_semantic_map(vis_feats: Tensor, txt_feats: Tensor):
        """Compute the semantic top down map using the VLM model (LSeg) from the top down vlmap"""
        # Compute the semantic labels and the semantic features on the map
        sem_labels = vlm.get_segmentation_labels(vis_feats, txt_feats)
        # Remove the empty cells
        sem_labels = np.where(mapper.vl_height_map.cpu().numpy() == 0.0, -1, sem_labels)

        # Compute the semantic map
        return get_semantic_mask_and_patch(sem_labels)

    def visualize_semantic_obs(vis_feats: Tensor, txt_feats: Tensor):
        """Compute the semantic top down map using the VLM model (LSeg) from the top down vlmap"""
        # Compute the semantic labels and the semantic features on the map
        sem_labels = vlm.get_segmentation_labels(vis_feats, txt_feats)

        # Compute the semantic map
        return get_semantic_mask_and_patch(sem_labels)

    # load offline data
    rgb_arr, depth_arr, pose_arr, data_num = load_offline_episode_data(
        cfg["DATASET_PATH"]
    )

    # Create the visual-langauge model
    vlm = Segmentation(cfg)

    # Create the mapper
    mapper = VisLangMapper(cfg)

    # Init the visual language mapper
    init_pose = np.expand_dims(pose_arr[0], axis=0)
    mapper.init_episode(init_pose)

    # Init for visualization
    fig, arr = plt.subplots(1, 4, figsize=(16, 12))
    artist_1, artist_2, artist_3, artist_4 = None, None, None, None

    # Save the last poses
    last_sim_pose = init_pose
    # Build map in a parallel manner
    for t in range(0, data_num):
        # load the data
        rgb_img = np.expand_dims(rgb_arr[t], axis=0)
        depth_img = np.expand_dims(depth_arr[t], axis=0)
        curr_sim_pose = np.expand_dims(pose_arr[t], axis=0)

        # Compute the visual-langauge feature
        rgb_tensor = torch.from_numpy(rgb_img).float().to(mapper.device)
        visual_feats = vlm.forward_visual_features(rgb_tensor)

        # Compute the textual feature
        textual_feats = vlm.forward_textual_features()

        # Visualize the semantic observation
        batch_sem_img, batch_img_patches = visualize_semantic_obs(
            visual_feats, textual_feats
        )

        # Pre-process the depth image
        depth_img = du.preprocess_depth(depth_img, cfg["VL_MAP"]["VISION_RANGE"])
        depth_tensor = to_tensor(depth_img, mapper.device)  # to tensor and GPU

        # Compute the rotation and translation matrices
        rot_mat, trans_mat, curr_map_pose = get_rot_and_trans_matrices(
            last_sim_pose, curr_sim_pose, mapper.curr_map_pose
        )

        episode_rot_tensor = to_tensor(rot_mat, mapper.device)
        episode_trans_tensor = to_tensor(trans_mat, mapper.device)

        # Update the last map poses and last sim poses
        last_sim_pose = curr_sim_pose
        mapper.curr_map_pose = curr_map_pose

        # Update the global map
        start_time = time.time()
        with torch.no_grad():
            # Update the maps
            batch_occ_map = mapper.update(
                visual_feats, depth_tensor, episode_rot_tensor, episode_trans_tensor
            )
            batch_occ_map = batch_occ_map.cpu().numpy()
            # Compute the semantic map
            batch_sem_map, batch_map_patch = visualize_semantic_map(
                mapper.vl_feat_map, textual_feats
            )

        # Plot the results
        fig.suptitle(f"Time step = {t} | seconds = {time.time() - start_time:.2f}")
        arr[3].legend(
            handles=batch_map_patch[0],
            loc="upper left",
            bbox_to_anchor=(1.0, 1),
            prop={"size": 9},
        )
        if t == 0:
            arr[0].set_title("Color Observation")
            artist_1 = arr[0].imshow(rgb_img[0])
            arr[1].set_title("Predicted Semantic Observation")
            artist_2 = arr[1].imshow(batch_sem_img[0])
            arr[2].set_title("Constructed Occupancy Map")
            artist_3 = arr[2].imshow(batch_occ_map[0])
            arr[3].set_title("Predicted Semantic Map")
            artist_4 = arr[3].imshow(batch_sem_map[0])
        else:
            artist_1.set_data(rgb_img[0])
            artist_2.set_data(batch_sem_img[0])
            artist_3.set_data(batch_occ_map[0])
            artist_4.set_data(batch_sem_map[0])
        fig.canvas.draw()
        plt.pause(0.001)

    debug.set_trace()


if __name__ == "__main__":
    """
    Run script in terminal using the command:
        PYTHONPATH=$PYTHON:./ python src/mappers/mapper_vlmaps.py

    1. This is a vlmaps using offline collected dataset. You can easily switch to online version by replacing
       76 - 78 with simulator interaction.
    2. Change the data path
    3. Download the checkpoint of LSeg from the official site (
        https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view
    )
    """
    # Load configuration
    configs = YamlParser("src/config/config_mapper_vlmaps.yaml").data
    configs["DATASET_PATH"] = "/home/xcg/research_projects/vln-ce-vlmap/data/Eudora"
    configs["MAP_SAVE_DIR"] = "tmp"

    # Build the map
    main(configs)
