from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import ImageColor
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from navsim.common.dataclasses import Camera, Lidar, Annotations, Trajectory
from navsim.common.enums import LidarIndex, BoundingBoxIndex
from navsim.visualization.config import AGENT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types


def add_camera_ax(ax: plt.Axes, camera: Camera) -> plt.Axes:
    """
    Adds camera image to matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :return: ax object with image
    """
    if camera.image is None: #加一个保护策略
        return ax
    ax.imshow(camera.image)
    return ax


def add_trajectory_to_camera_ax(
    ax: plt.Axes,
    camera: Camera,
    trajectory: Trajectory,
    which: str = "agent",
) -> plt.Axes:
    """Overlay a ground-plane trajectory onto a camera image.

    Assumes the trajectory is expressed in the ego / lidar frame at z=0.
    """

    if camera.image is None:
        return ax

    if (
        camera.sensor2lidar_rotation is None
        or camera.sensor2lidar_translation is None
        or camera.intrinsics is None
    ):
        # Missing calibration for this camera; just draw the raw image.
        ax.imshow(camera.image)
        return ax

    image = camera.image.copy() #每次都是新copy，所以都是新图！
    img_h, img_w = image.shape[:2]

    # Build 3D points on ground plane in ego/lidar frame: (x, y, 0).
    # Scene.get_future_trajectory() / agent.compute_trajectory() return poses
    # *excluding* the current ego position. To mimic BEV plotting behavior
    # (which prepends [0, 0]), we explicitly add the origin so the line
    # connects back to the vehicle center in the image.

    # 之前：只用未来点
    poses_xy = trajectory.poses[:, :2]
    z = np.zeros((poses_xy.shape[0], 1), dtype=np.float32)
    points_lidar = np.concatenate([poses_xy.astype(np.float32), z], axis=1)

    # 现在：先加上原点，再拼未来点
    # poses_xy = trajectory.poses[:, :2].astype(np.float32)
    # origin_xy = np.zeros((1, 2), dtype=np.float32)
    # poses_xy = np.concatenate([origin_xy, poses_xy], axis=0)
    # z = np.zeros((poses_xy.shape[0], 1), dtype=np.float32)
    # points_lidar = np.concatenate([poses_xy, z], axis=1)

    # Reuse the lidar->camera projection logic from _transform_pcs_to_images.
    sensor2lidar_rotation = camera.sensor2lidar_rotation
    sensor2lidar_translation = camera.sensor2lidar_translation

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4, dtype=np.float32)
    intrinsic = camera.intrinsics
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pts = np.concatenate([points_lidar, np.ones_like(points_lidar)[:, :1]], axis=1)
    cur_pts_cam = (lidar2img_rt @ cur_pts.T).T

    # eps = 1e-3
    # in_front = cur_pts_cam[:, 2] > eps
    # cur_pts_cam = cur_pts_cam[..., 0:2] / np.maximum(
    #     cur_pts_cam[..., 2:3], np.ones_like(cur_pts_cam[..., 2:3]) * eps
    # )

    # # Filter to image bounds.
    # x, y = cur_pts_cam[:, 0], cur_pts_cam[:, 1]
    # in_img = (
    #     in_front
    #     & (x > 0)
    #     & (x < img_w - 1)
    #     & (y > 0)
    #     & (y < img_h - 1)
    # )

    # pts_img = np.stack([x[in_img], y[in_img]], axis=1).astype(np.int32)
    # if len(pts_img) < 2:
    #     ax.imshow(image)
    #     return ax

    # cfg = TRAJECTORY_CONFIG.get(which, TRAJECTORY_CONFIG["agent"])
    # color = ImageColor.getcolor(cfg["line_color"], "RGB")

    # # Draw as a polyline.
    # for i in range(len(pts_img) - 1):
    #     cv2.line(
    #         image,
    #         (pts_img[i, 0], pts_img[i, 1]),
    #         (pts_img[i + 1, 0], pts_img[i + 1, 1]),
    #         color,
    #         int(cfg["line_width"]),
    #         cv2.LINE_AA,
    #     )


    eps = 1e-3
    depths = cur_pts_cam[:, 2]
    in_front = depths > eps
    cur_pts_cam = cur_pts_cam[..., 0:2] / np.maximum(
        depths[:, None], np.ones_like(depths[:, None]) * eps
    )

    x, y = cur_pts_cam[:, 0], cur_pts_cam[:, 1]

    cfg = TRAJECTORY_CONFIG.get(which, TRAJECTORY_CONFIG["agent"])
    color = ImageColor.getcolor(cfg["line_color"], "RGB")

    # For each consecutive pair, use cv2.clipLine to keep the in-image
    # portion of the segment even if one or both endpoints lie outside
    # the image bounds.
    roi = (0, 0, img_w - 1, img_h - 1)
    num_pts = len(x)
    if num_pts < 2:
        return image

    for i in range(num_pts - 1):
        # Skip if both points are behind the camera.
        if not (in_front[i] or in_front[i + 1]):
            continue

        p1 = (float(x[i]), float(y[i]))
        p2 = (float(x[i + 1]), float(y[i + 1]))

        ok, clipped_p1, clipped_p2 = cv2.clipLine(
            roi,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
        )
        if not ok:
            continue

        cv2.line(
            image,
            clipped_p1,
            clipped_p2,
            color,
            int(cfg["line_width"]),
            cv2.LINE_AA,
        )

        
    ax.imshow(image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: Camera, lidar: Lidar) -> plt.Axes:
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """

    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    for (x, y), color in zip(pc_in_cam[pc_in_fov_mask], lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax


def add_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with bounding boxes on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)

    ax.imshow(image)
    return ax


def _transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Helper function to transform bounding boxes into camera frame
    TODO: Refactor
    :param boxes: array representation of bounding boxes
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :return: bounding boxes in camera coordinates
    """

    locs, rots = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.HEADING :],
    )
    dims_cam = boxes[
        :, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot = Quaternion(axis=[0, 0, 1], radians=rot)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return np.concatenate([locs_cam, dims_cam, rots_cam], -1)


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """
    Rotate 3D points by angles according to axis.
    TODO: Refactor
    :param points: array of points
    :param angles: array of angles
    :param axis: axis to perform rotation, defaults to 0
    :raises value: _description_
    :raises ValueError: if axis invalid
    :return: rotated points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.float32],
    box_corners: npt.NDArray[np.float32],
    box_labels: List[str],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """
    Plot the boundary lines of 3D rectangular on 2D images.
    TODO: refactor
    :param image:  The numpy array of image.
    :param box_corners: Coordinates of the corners of 3D, shape of [N, 8, 2].
    :param box_labels: labels of boxes for coloring
    :param thickness: pixel width of liens, defaults to 3
    :return: image with 3D bounding boxes
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        layer = tracked_object_types[box_labels[i]]
        color = ImageColor.getcolor(AGENT_CONFIG[layer]["fill_color"], "RGB")
        corners = box_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _transform_pcs_to_images(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov
