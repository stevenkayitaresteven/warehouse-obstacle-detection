"""
Obstacle fusion module.

Associates YOLO object detections with regions of a relative-depth map.
"""

import numpy as np
import cv2


def fuse_detections_with_depth(detections, depth_map):
    """
    Associate each detected object with its median relative depth.

    Parameters
    ----------
    detections : list of dict
        Each detection must contain:
        - class_name
        - confidence
        - bbox: (x1, y1, x2, y2)

    depth_map : numpy.ndarray
        Two-dimensional relative-depth map.

    Returns
    -------
    list of dict
        Detected objects enriched with depth information.
    """

    depth_map = np.asarray(depth_map)

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D array.")

    height, width = depth_map.shape
    obstacles = []

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        # Convert bounding-box coordinates to integers
        x1 = int(np.floor(x1))
        y1 = int(np.floor(y1))
        x2 = int(np.ceil(x2))
        y2 = int(np.ceil(y2))

        # Clip coordinates to image boundaries
        x1 = int(np.clip(x1, 0, width))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height))
        y2 = int(np.clip(y2, 0, height))

        # Extract depth values inside the bounding box
        depth_region = depth_map[y1:y2, x1:x2]

        # Keep only finite depth values
        valid_depths = depth_region[np.isfinite(depth_region)]

        if valid_depths.size == 0:
            median_depth = None
        else:
            median_depth = float(np.median(valid_depths))

        obstacle = {
            "class_name": detection["class_name"],
            "confidence": float(detection["confidence"]),
            "bbox": (x1, y1, x2, y2),
            "relative_depth": median_depth,
            "valid_depth_pixels": int(valid_depths.size)
        }

        obstacles.append(obstacle)

    return obstacles


def create_driving_corridor_mask(
    image_shape,
    top_y_ratio=0.55,
    top_width_ratio=0.30,
    bottom_width_ratio=0.90
):
    """
    Create a trapezoidal region of interest in front of the robot.

    Parameters
    ----------
    image_shape : tuple
        (height, width) of the image.

    top_y_ratio : float
        Vertical position where the corridor begins.

    top_width_ratio : float
        Corridor width near its top, relative to image width.

    bottom_width_ratio : float
        Corridor width at the bottom, relative to image width.

    Returns
    -------
    numpy.ndarray
        Binary mask: 255 inside the corridor, 0 outside.
    """

    height, width = image_shape[:2]

    if not 0 <= top_y_ratio < 1:
        raise ValueError("top_y_ratio must be between 0 and 1.")

    if not 0 < top_width_ratio <= 1:
        raise ValueError("top_width_ratio must be between 0 and 1.")

    if not 0 < bottom_width_ratio <= 1:
        raise ValueError("bottom_width_ratio must be between 0 and 1.")

    if top_width_ratio > bottom_width_ratio:
        raise ValueError("Top width cannot exceed bottom width.")

    center_x = width // 2

    top_y = int(height * top_y_ratio)

    top_half_width = int(width * top_width_ratio / 2)
    bottom_half_width = int(width * bottom_width_ratio / 2)

    polygon = np.array([
        [center_x - top_half_width, top_y],
        [center_x + top_half_width, top_y],
        [center_x + bottom_half_width, height - 1],
        [center_x - bottom_half_width, height - 1]
    ], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [polygon], 255)

    return mask


def create_uncovered_candidate_mask(corridor_mask, detections):
    """
    Identify corridor pixels not covered by known YOLO detections.

    These pixels are candidates for further analysis, NOT confirmed
    unknown obstacles.
    """

    corridor_mask = np.asarray(corridor_mask)

    if corridor_mask.ndim != 2:
        raise ValueError("corridor_mask must be a 2D array.")

    height, width = corridor_mask.shape

    # Convert corridor into a boolean mask
    corridor = corridor_mask > 0

    # Create mask for known detected objects
    known_mask = np.zeros((height, width), dtype=bool)

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        x1 = int(np.clip(np.floor(x1), 0, width))
        x2 = int(np.clip(np.ceil(x2), 0, width))
        y1 = int(np.clip(np.floor(y1), 0, height))
        y2 = int(np.clip(np.ceil(y2), 0, height))

        if x2 > x1 and y2 > y1:
            known_mask[y1:y2, x1:x2] = True

    # Keep corridor pixels not covered by known detections
    candidate_mask = corridor & ~known_mask

    return candidate_mask.astype(np.uint8) * 255
