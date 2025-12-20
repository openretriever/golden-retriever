import numpy as np
import torch
from torch import Tensor

import src.mappers.utils_vlmaps.utils.depth_utils as du


class VisLangMapper:
    def __init__(self, cfg) -> None:
        """Visual-language Mapper"""
        # CUDA device
        self.device = torch.device(cfg["VLM_MODEL"]["DEVICE"])
        # batch size
        self.batch_size = cfg["VL_MAP"]["BATCH_SIZE"]

        """ Camera configuration """
        self.frame_width = cfg["VL_MAP"]["CAMERA"]["FRAME_WIDTH"]  # image width (int)
        self.frame_height = cfg["VL_MAP"]["CAMERA"][
            "FRAME_HEIGHT"
        ]  # image height (int)
        # camera field of view in degrees (float)
        self.camera_fov = cfg["VL_MAP"]["CAMERA"]["FOV"]
        self.camera_height = cfg["VL_MAP"]["CAMERA"][
            "HEIGHT"
        ]  # camera height in cms (float)
        # camera tilt angle in degrees (float)
        self.camera_tilt_angle = cfg["VL_MAP"]["CAMERA"]["TILT_ANGLE"]
        # camera intrinsic matrix
        self.camera_matrix = du.get_camera_matrix(
            self.frame_width, self.frame_height, self.camera_fov
        ).to(self.device)

        """ Map configuration """
        # map size: 2400 x 2400 cm^2 (24 x 24 m^2) (int)
        self.map_size_cm = cfg["VL_MAP"]["SIZE"]
        # map resolution: 25 cm^2 per cell (int)
        self.map_resolution_cm = cfg["VL_MAP"]["RESOLUTION"]
        # map size: grid num (int)
        self.map_size_grid = round(self.map_size_cm / self.map_resolution_cm)
        self.map_buffer = None  # occupancy map

        # visual-language feature dimension
        self.vl_feat_dim = cfg["VLM_MODEL"]["CLIP"]["FEAT_DIM"]
        self.vl_feat_map = None  # Each cell stores the visual feature
        self.vl_count_map = None  # Each cell stores the count of the visual feature
        # Each cell stores the height (highest) of the visual feature
        self.vl_height_map = None

        self.pc_scale = cfg["VL_MAP"]["CAMERA"]["PC_SCALE"]  # point could sample ratio
        self.z_bins = torch.tensor(
            [
                cfg["VL_MAP"]["OBSTACLE_HEIGHT"]["MIN"],
                cfg["VL_MAP"]["OBSTACLE_HEIGHT"]["MAX"],
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.obs_threshold = 1

        # Pose
        self.last_sim_pose = None  # pose in [meter, meter, radius]
        self.curr_sim_pose = None  # pose in [meter, meter, radius]
        self.curr_map_pose = None  # pose in [meter, meter, degree]

        self.init_map_pose = np.tile(
            [self.map_size_cm / 100.0 / 2.0, self.map_size_cm / 100.0 / 2.0, 0.0],
            (self.batch_size, 1),
        )

    def get_map_rel_pose(self, poses: np.ndarray) -> np.ndarray:
        """
        get map pose in (x,y) each in range(0,1)
        # here x,y is conventional image axis

        args:
            poses: map pose (B,3)
        """
        xy = poses[:, :2] * 100 / self.map_size_cm  # b,2
        return xy

    def init_episode(self, init_poses: np.ndarray) -> None:
        """Initialize the map and the agents's pose
        The map convention:
            - Origin: top-left
            - Positive X: to the right
            - Positive Y: to the downside
        """
        # Init visual language feature map
        self.vl_feat_map = torch.zeros(
            (
                self.batch_size,
                self.map_size_grid,
                self.map_size_grid,
                self.vl_feat_dim,
            ),
            dtype=torch.float,
            device=self.device,
        )
        # Init visual language count map
        self.vl_count_map = torch.zeros(
            (self.batch_size, self.map_size_grid, self.map_size_grid),
            dtype=torch.long,
            device=self.device,
        )
        # Init visual language height map
        self.vl_height_map = torch.zeros(
            (self.batch_size, self.map_size_grid, self.map_size_grid),
            dtype=torch.float,
            device=self.device,
        )

        # Init occupancy map
        self.map_buffer = torch.zeros(
            (
                self.batch_size,
                self.map_size_grid,
                self.map_size_grid,
                len(self.z_bins) + 1,
            ),
            dtype=torch.float,
            device=self.device,
        )

        # Initialize the sim poses
        self.last_sim_pose = init_poses
        self.curr_sim_pose = init_poses

        # By default, the robot is initialized in the middle of the map, facing to the right.
        self.curr_map_pose = self.init_map_pose.copy()

    def update(
        self, color_feat: Tensor, depth: Tensor, rot_mat: Tensor, trans_mat: Tensor
    ) -> Tensor:
        """
        Update the Visual-language Map using the latest observations
        Args:
            color_feat (B x H x W x D): visual-language features default feature dimension D = 512
            depth (B x H x W): depth images
            rot_mat (B x 3 x 3): rotation matrix
            trans_mat (B x 2): transition matrix

        Returns:
            global_occ (B x H_map x W_map): global occupancy map

        """
        """ Down-sample the rgb features """
        color_feat = color_feat[:, :: self.pc_scale, :: self.pc_scale]

        """ Back-project the depth to point cloud and down-sample the point cloud """
        # Egocentric camera frame: X (right), Y (inside), and Z (up)
        point_cloud = du.get_point_cloud(depth, self.camera_matrix, self.pc_scale)

        # Raise to camera height
        camera_view = du.transform_camera_view(
            point_cloud, self.camera_height, self.camera_tilt_angle
        )

        # Transform to global map frame: X (right), Y (down), Origin (Top left)
        # The global map frame is LEFT-HANDED
        geocentric_pc = du.transform_pose(camera_view, rot_mat, trans_mat)

        """ Update the global occupancy map """
        # Project the point cloud into the xy-z plane (global)
        geocentric_view_flat = du.bin_points(
            geocentric_pc, self.map_size_grid, self.z_bins, self.map_resolution_cm
        )

        # Compute the occupancy map
        self.map_buffer += geocentric_view_flat
        global_occ = self.map_buffer[..., 1] / self.obs_threshold
        global_occ[global_occ >= 0.5] = 1.0
        global_occ[global_occ < 0.5] = 0.0

        """ Update the global visual feature map """
        self.vl_feat_map, self.vl_count_map, self.vl_height_map = du.bin_features(
            geocentric_pc,
            color_feat,
            self.map_size_grid,
            self.z_bins,
            self.map_resolution_cm,
            self.vl_feat_map,
            self.vl_count_map,
            self.vl_height_map,
        )

        return global_occ
