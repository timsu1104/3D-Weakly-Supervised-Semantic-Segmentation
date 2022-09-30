"""
Modified from rendering module of pi-GAN. 
"""

import math
import numpy as np
import random
from typing import List

def normalize_vecs(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vector lengths.
    """
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True))

def sample_camera_pose(BatchSize: int, radius=2.7, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal', intrinsics: List[int] =None, cfg: str = 'Shapenet') -> np.ndarray:
    """Samples n camera positions."""

    camera_origin, pitch, yaw = sample_camera_positions(n=BatchSize, r=radius, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin)

    if intrinsics is None:
        # focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
        # intrinsics = np.array([focal_length, 0, 0.5, 0, focal_length, 0.5, 0, 0, 1], dtype=np.float32)
        intrinsics = np.array([525.0, 0.0, 256.0, 0.0, 525.0, 256.0, 0.0, 0.0, 100.0], dtype=np.float32) / 100.0
    intrinsics = np.repeat(intrinsics[None, :], BatchSize, 0)
    
    cam_pose = np.concatenate([cam2world_matrix.reshape(BatchSize, 16), intrinsics], axis=1, dtype=np.float32)

    return cam_pose, pitch, yaw

def sample_camera_positions(n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    if mode == 'uniform':
        theta = (np.random.rand(n, 1) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (np.random.rand(n, 1) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = np.random.randn(n, 1) * horizontal_stddev + horizontal_mean
        phi = np.random.randn(n, 1) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (np.random.rand(n, 1) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (np.random.rand(n, 1) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = np.random.randn(n, 1) * horizontal_stddev + horizontal_mean
            phi = np.random.randn(n, 1) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (np.random.rand(n, 1) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((np.random.rand(n, 1) - .5) * 2 * v_stddev + v_mean)
        v = np.clip(v, 1e-5, 1 - 1e-5)
        phi = np.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = np.ones((n, 1), dtype=np.float32) * horizontal_mean
        phi = np.ones((n, 1), dtype=np.float32) * vertical_mean

    phi = np.clip(phi, 1e-5, math.pi - 1e-5)

    output_points = np.zeros((n, 3))
    # output_points[:, 0:1] = r*np.sin(phi) * np.cos(theta)
    # output_points[:, 2:3] = r*np.sin(phi) * np.sin(theta)
    # output_points[:, 1:2] = r*np.cos(phi)
    output_points[:, 0:1] = r*np.sin(phi) * np.cos(theta)
    output_points[:, 1:2] = r*np.sin(phi) * np.sin(theta)
    output_points[:, 2:3] = r*np.cos(phi)

    return output_points, phi, theta

def create_cam2world_matrix(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = np.broadcast_to(np.array([[0, 1, 0]], dtype=np.float32), forward_vector.shape)

    left_vector = normalize_vecs(np.cross(up_vector, forward_vector, axis=-1))

    up_vector = normalize_vecs(np.cross(forward_vector, left_vector, axis=-1))

    rotation_matrix = np.eye(4)[None].repeat(forward_vector.shape[0], 0)
    rotation_matrix[:, :3, :3] = np.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = np.eye(4)[None].repeat(forward_vector.shape[0], 0)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

if __name__ == '__main__':
    
    print("Testing sampler...... ")
    # sample_camera_pose(2, mode='uniform')
    # sample_camera_pose(2, mode='normal')
    # sample_camera_pose(2, mode='hybrid')
    # sample_camera_pose(2, mode='truncated_gaussian')
    s, _, _ = sample_camera_pose(2, mode='spherical_uniform', radius=1.3)
    # sample_camera_pose(2, mode='other')
    print(s.shape)
    print("Test passed. Spherical uniformed sample:", np.linalg.norm(s[0,:16].reshape(4, 4)[:3, 3]))