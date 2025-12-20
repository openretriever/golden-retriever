""" Utilities for processing depth images using tensor
"""
import numpy as np
import torch

ANGLE_EPS = 0.001


def normalize(v):
    return v / np.linalg.norm(v)


def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32,
        )
        R = (
            np.eye(3)
            + np.sin(angle) * S_hat
            + (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
        )
    else:
        R = np.eye(3)
    return R


def r_between(v_from_, v_to_):
    v_from = normalize(v_from_)
    v_to = normalize(v_to_)
    ax = normalize(np.cross(v_from, v_to))
    angle = np.arccos(np.dot(v_from, v_to))
    return get_r_matrix(ax, angle)


def rotate_camera_to_point_at(up_from, lookat_from, up_to, lookat_to):
    inputs = [up_from, lookat_from, up_to, lookat_to]
    for i in range(4):
        inputs[i] = normalize(np.array(inputs[i]).reshape((-1,)))
    up_from, lookat_from, up_to, lookat_to = inputs
    r1 = r_between(lookat_from, lookat_to)

    new_x = np.dot(r1, np.array([1, 0, 0]).reshape((-1, 1))).reshape((-1))
    to_x = normalize(np.cross(lookat_to, up_to))
    angle = np.arccos(np.dot(new_x, to_x))
    if angle > ANGLE_EPS:
        if angle < np.pi - ANGLE_EPS:
            ax = normalize(np.cross(new_x, to_x))
            flip = np.dot(lookat_to, ax)
            if flip > 0:
                r2 = get_r_matrix(lookat_to, angle)
            elif flip < 0:
                r2 = get_r_matrix(lookat_to, -1.0 * angle)
        else:
            # Angle of rotation is too close to 180 degrees, direction of rotation
            # does not matter.
            r2 = get_r_matrix(lookat_to, angle)
    else:
        r2 = np.eye(3)
    return np.dot(r2, r1)


def preprocess_depth(depth, agent_view_range):
    """Process the depth image."""
    # Remove depth > 9.9 meters
    mask2 = depth > 9.9
    depth[mask2] = 0.0

    # Convert depth == 0.0 to NaN
    for i in range(depth.shape[1]):
        depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()
    mask1 = depth == 0
    depth[mask1] = np.NaN

    # Convert depth from m to cm
    depth *= 100.0
    with np.errstate(invalid="ignore"):
        depth[depth > agent_view_range] = np.NaN
    return depth


def get_camera_matrix(width: int, height: int, fov: float):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
    camera_matrix_tensor = torch.tensor([xc, zc, f], dtype=torch.float)
    return camera_matrix_tensor


def get_point_cloud(Y, camera_matrix, scale):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = torch.meshgrid(
        torch.arange(Y.shape[-1]), torch.arange(Y.shape[-2] - 1, -1, -1)
    )
    x, z = x.T, z.T  # The default indexing is "ij" in torch.meshgrid

    x = x.unsqueeze(dim=0).to(Y.device)
    z = z.unsqueeze(dim=0).to(Y.device)

    # Sample the depth
    X = (
        (x[:, ::scale, ::scale] - camera_matrix[0])
        * Y[:, ::scale, ::scale]
        / camera_matrix[2]
    )
    Z = (
        (z[:, ::scale, ::scale] - camera_matrix[1])
        * Y[:, ::scale, ::scale]
        / camera_matrix[2]
    )

    # Construct the point cloud
    XYZ = torch.cat(
        [
            X.unsqueeze(dim=-1),
            Y[:, ::scale, ::scale].unsqueeze(dim=-1),
            Z.unsqueeze(dim=-1),
        ],
        dim=-1,
    )
    return XYZ


def transform_camera_view(XYZ: torch.tensor, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ  (H x W x 3)        : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    # R shape: 3 x 3: R is identity if camera_elevation_degree = 0.0
    R = torch.tensor(
        get_r_matrix([1.0, 0.0, 0.0], angle=np.deg2rad(camera_elevation_degree)),
        dtype=torch.float,
        device=XYZ.device,
    )
    # Rotate the positions
    XYZ = torch.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    # Raise the camera height
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, R, T):
    # Perform batch matrix multiplication
    sh = XYZ.shape
    XYZ = torch.bmm(XYZ.reshape(XYZ.shape[0], -1, 3), R)
    XYZ[..., 0] = XYZ[..., 0] + T[..., 0].reshape(-1, 1)
    XYZ[..., 1] = XYZ[..., 1] + T[..., 1].reshape(-1, 1)
    return XYZ.reshape(sh)


def bin_features(
    XYZ_cms,
    XYZ_colors,
    map_size,
    z_bins,
    xy_resolution,
    feature_maps,
    count_maps,
    height_maps,
):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    n_z_bins = len(z_bins) + 1
    vl_feature_maps = []
    vl_count_maps = []
    vl_height_maps = []
    for XYZ_cm, XYZ_color, feat_map, count_map, height_map in zip(
        XYZ_cms, XYZ_colors, feature_maps, count_maps, height_maps, strict=False
    ):
        """Find out all invalid depth values (out of range)"""
        isnotnan = torch.logical_not(torch.isnan(XYZ_cm[:, :, 0]))

        """ Discretization of the XY plane """
        X_bin = torch.round(XYZ_cm[:, :, 0] / xy_resolution).long()
        Y_bin = torch.round(XYZ_cm[:, :, 1] / xy_resolution).long()

        """ Discretization of the Z axis (height) """
        # Note that:
        #   - value > max z value: len(z_bins)
        #   - value < min z value: 0
        Z_bin = torch.bucketize(XYZ_cm[:, :, 2], boundaries=z_bins).long()

        """ Mark each point in the point cloud with True/False for valid/invalid point """
        isvalid = torch.stack(
            [
                X_bin >= 0,
                X_bin < map_size,
                Y_bin >= 0,
                Y_bin < map_size,
                Z_bin >= 0,
                Z_bin < n_z_bins - 1,
                isnotnan,
            ]
        )
        isvalid = torch.all(isvalid, dim=0)

        """ Save the feature into the map """
        # update the visual features into the map
        row, col, _ = XYZ_cm.shape

        # Find indices of all valid poses
        pose_indices = torch.where(isvalid == True)
        # Get the corresponding bin indices for all poses
        y_bin_indices = Y_bin[pose_indices]
        x_bin_indices = X_bin[pose_indices]

        # Get the count map for selected bins
        count = count_map[y_bin_indices, x_bin_indices]
        # Get the height map for selected bins
        height = height_map[y_bin_indices, x_bin_indices]

        """ Assign value to empty cells """
        # Compute the mask to select all empty cells from the selected poses
        empty_cell_mask = torch.logical_and(
            count == 0, height == 0
        )  # count == 0 and height == 0
        if True in empty_cell_mask:  # If there exists unassigned cells
            # Find the bin indices for both Y bin and X bin
            y_bin_empty_indices = y_bin_indices[empty_cell_mask]
            x_bin_empty_indices = x_bin_indices[empty_cell_mask]

            i_pose_empty_indices = pose_indices[0][empty_cell_mask]
            j_pose_empty_indices = pose_indices[1][empty_cell_mask]
            curr_empty_feat = XYZ_color[i_pose_empty_indices, j_pose_empty_indices]
            curr_empty_height = XYZ_cm[i_pose_empty_indices, j_pose_empty_indices][
                ..., 2
            ]
            # Assign value to all empty cells
            feat_map[y_bin_empty_indices, x_bin_empty_indices] = curr_empty_feat
            height_map[y_bin_empty_indices, x_bin_empty_indices] = curr_empty_height
            count_map[y_bin_empty_indices, x_bin_empty_indices] += 1

        """ Assign value to occupied cells """
        # Compute the mask for occupied cells
        occupied_cell_mask = torch.logical_or(count != 0, height != 0)
        if True in occupied_cell_mask:
            # Find all occupied cells
            y_bin_occupied_indices = y_bin_indices[occupied_cell_mask]
            x_bin_occupied_indices = x_bin_indices[occupied_cell_mask]
            occupied_height = height_map[y_bin_occupied_indices, x_bin_occupied_indices]

            i_pose_occupied_indices = pose_indices[0][occupied_cell_mask]
            j_pose_occupied_indices = pose_indices[1][occupied_cell_mask]
            curr_height = XYZ_cm[i_pose_occupied_indices, j_pose_occupied_indices][
                ..., 2
            ]

            # Find higher cells
            higher_cell_mask = curr_height > occupied_height
            if True in higher_cell_mask:
                y_bin_occupied_higher_indices = y_bin_occupied_indices[higher_cell_mask]
                x_bin_occupied_higher_indices = x_bin_occupied_indices[higher_cell_mask]

                i_pose_occupied_higher_indices = i_pose_occupied_indices[
                    higher_cell_mask
                ]
                j_pose_occupied_higher_indices = j_pose_occupied_indices[
                    higher_cell_mask
                ]

                # Replace the lower cell with the higher cell
                feat_map[
                    y_bin_occupied_higher_indices, x_bin_occupied_higher_indices
                ] = XYZ_color[
                    i_pose_occupied_higher_indices, j_pose_occupied_higher_indices
                ]
                height_map[
                    y_bin_occupied_higher_indices, x_bin_occupied_higher_indices
                ] = XYZ_cm[
                    i_pose_occupied_higher_indices, j_pose_occupied_higher_indices
                ][
                    ..., 2
                ]
                count_map[
                    y_bin_occupied_higher_indices, x_bin_occupied_higher_indices
                ] = 1

            # Aggregate the feature with the same height
            same_cell_mask = curr_height == occupied_height
            if True in same_cell_mask:
                y_bin_occupied_same_indices = y_bin_occupied_indices[same_cell_mask]
                x_bin_occupied_same_indices = x_bin_occupied_indices[same_cell_mask]

                i_pose_occupied_same_indices = i_pose_occupied_indices[same_cell_mask]
                j_pose_occupied_same_indices = j_pose_occupied_indices[same_cell_mask]

                feat_map[y_bin_occupied_same_indices, x_bin_occupied_same_indices] += (
                    XYZ_color[
                        i_pose_occupied_same_indices, j_pose_occupied_same_indices
                    ]
                    - feat_map[y_bin_occupied_same_indices, x_bin_occupied_same_indices]
                ) / (
                    count_map[y_bin_occupied_same_indices, x_bin_occupied_same_indices]
                    + 1
                ).reshape(
                    -1, 1
                )
                count_map[y_bin_occupied_same_indices, x_bin_occupied_same_indices] += 1

        # Update the maps
        vl_feature_maps.append(feat_map)
        vl_height_maps.append(height_map)
        vl_count_maps.append(count_map)

    return (
        torch.stack(vl_feature_maps),
        torch.stack(vl_count_maps),
        torch.stack(vl_height_maps),
    )


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    for XYZ_cm in XYZ_cms:
        isnotnan = torch.logical_not(torch.isnan(XYZ_cm[:, :, 0]))
        X_bin = torch.round(XYZ_cm[:, :, 0] / xy_resolution)
        Y_bin = torch.round(XYZ_cm[:, :, 1] / xy_resolution)
        Z_bin = torch.bucketize(XYZ_cm[:, :, 2], boundaries=z_bins)

        isvalid = torch.stack(
            [
                X_bin >= 0,
                X_bin < map_size,
                Y_bin >= 0,
                Y_bin < map_size,
                Z_bin >= 0,
                Z_bin < n_z_bins,
                isnotnan,
            ]
        )
        isvalid = torch.all(isvalid, dim=0)

        # This purely compute the index for each point cloud point
        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        # Set all invalid locations to have the index = 0
        ind[torch.logical_not(isvalid)] = 0
        # Count for each valid index
        count = torch.bincount(
            ind.flatten().int(),
            isvalid.flatten().int(),
            minlength=map_size * map_size * n_z_bins,
        )
        counts.append(count.reshape([map_size, map_size, n_z_bins]))

    counts = torch.stack(counts, dim=0)

    return counts
