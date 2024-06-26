
import cv2
import matplotlib.pyplot as plt

import numpy as np
from dataclasses import dataclass
from matplotlib.axes import Axes
from tqdm import tqdm
import h5py

from scipy.spatial.transform import Rotation as Rot

import copy
import re
import os
from pathlib import Path

from ImageMatcher import ImagePair


def load_video(path: str, grayscale: bool = True,
               frame_segment: tuple[int, int] = (0, 99999)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []

    for i in range(0, frame_segment[0] + frame_segment[1]):

        ret, frame = cap.read()
        if not ret:
            break

        if i < frame_segment[0]:
            #frames.append(None)
            continue

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale else frame)

    cap.release()
    return frames


def project_points(H, pts, keep_z=False):

    assert H.shape == (3, 3)
    assert len(pts.shape) == 2 and pts.shape[1] in [2, 3]
    if pts.shape[1] == 2:
        o = np.ones([pts.shape[0], 1])
        pts_h = np.concatenate([pts, o], axis=1)
    else:
        pts_h = pts
    pts_p = pts_h @ H.T
    if keep_z:
        return pts_p
    return pts_p[:, :2] / pts_p[:, 2:3]


def reprojection_error(H, src_pts, dst_pts, plot=False):

    proj_pts = project_points(H, src_pts)

    if plot:
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], marker='x', c='r', lw=0.5)
        plt.scatter(dst_pts[:, 0], dst_pts[:, 1], marker='+', c='g', lw=1)
        plt.show()

    return np.sqrt(np.square(proj_pts - dst_pts).sum(axis=1)).mean()


def visualize_matches(pair: ImagePair, dpi=100, masks=None):

    axs: tuple[Axes]
    fig, axs = plt.subplots(1, 2, dpi=dpi)
    a1, a2 = axs

    for a in axs:
        a.set_axis_off()

    a1.imshow(pair.img1, 'gray')
    if masks is not None:
        a1.imshow(masks[pair.i], 'jet', alpha=0.4)
    a1.scatter(*pair.src_pts.T, c='lime', marker='.', s=1, lw=1)

    a2.imshow(pair.img2, 'gray')
    if masks is not None:
        a2.imshow(masks[pair.j], 'jet', alpha=0.4)
    a2.scatter(*pair.dst_pts.T, c='lime', marker='.', s=1, lw=1)

    fig.tight_layout()
    plt.show()

def import_image_pairs(path: str, frames: list[np.ndarray]) -> list[ImagePair]:
    pairs = []
    with h5py.File(path, 'r') as file:
        n_pairs = file.attrs['n_pairs']
        for i in range(n_pairs):
            g = file[f'pair_{i}']
            pairs.append(ImagePair(
                frames[g.attrs['i']],
                frames[g.attrs['j']],
                np.array(g['H']),
                np.array(g['src_pts']),
                np.array(g['dst_pts']),
                g.attrs['i'],
                g.attrs['j'],
                g.attrs['still'],
            ))
    return pairs

def export_image_pairs(path: str, ps: list[ImagePair]):
    with h5py.File(path, 'w') as f:
        for idx, p in enumerate(ps):
            g = f.create_group(f'pair_{idx}')
            g.create_dataset('H', data=p.H)
            g.create_dataset('src_pts', data=p.src_pts)
            g.create_dataset('dst_pts', data=p.dst_pts)
            g.attrs['i'] = p.i
            g.attrs['j'] = p.j
            g.attrs['still'] = p.still

        indices = {}
        for p in ps:
            if p.i not in indices:
                indices[p.i] = 1
            if p.j not in indices:
                indices[p.j] = 1

        f.attrs['n_pairs'] = len(ps)
        f.create_dataset('cam_indices', data=sorted(indices.keys()))

def import_cam_params(path: str) -> dict[np.ndarray]:
    cam_params = {}
    with h5py.File(path, 'r') as f:
        for d in f.values():
            cam_idx = d.attrs['cam_idx']
            cam_params[cam_idx] = np.array(d)
    return cam_params

def load_est_gt_poses(vid_path, vid_w):

    vid_path = Path(vid_path)
    data_dir = vid_path.parent
    vid_stem = vid_path.stem

    cam_params_raw = import_cam_params(
        os.path.join(data_dir, vid_stem) + '-opt-poses.h5')

    cam_params_est = {}
    for i, p in sorted(cam_params_raw.items()):
        cam_params_est[i] = {
            'R': Rot.from_rotvec(p[:3], degrees=False).as_matrix(),
            'hfov': 2 * np.degrees(np.arctan(vid_w * 0.5 / p[3])),
        }

    gt_path = os.path.join(data_dir, f'{vid_stem[:4]}.npy')

    if not os.path.isfile(gt_path):
        return cam_params_est, None

    gt = np.load(gt_path)
    cam_params_gt = {}
    for i in sorted(cam_params_est.keys()):
        p = gt[i]
        cam_params_gt[i] = {
            'R': Rot.from_euler('YXZ', p[:3], degrees=True).as_matrix(),
            'hfov': p[3],
        }

    return cam_params_est, cam_params_gt

def set_ref_cam(ref_cam_idx, cam_params_est, cam_params_gt):

    cpe = copy.deepcopy(cam_params_est)
    cpg = copy.deepcopy(cam_params_gt)

    R = cpe[ref_cam_idx]['R'].T.copy()
    for p in cpe.values():
        p['R'] = R @ p['R']

    R = cpg[ref_cam_idx]['R'].T.copy()
    for p in cpg.values():
        p['R'] = R @ p['R']

    return cpe, cpg

def hfov_to_K(hfov_degrees, w, h):
    f = 0.5 * w / np.tan(np.radians(hfov_degrees / 2))
    K = np.array([[-f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    return K

def view_to_world(p, w, h):
    R = p['R']
    K = hfov_to_K(p['hfov'], w, h)
    return R @ np.linalg.inv(K)

def H_between_frames(p0, p1, w, h):
    """ Compute transformation H from image plane of p0 to p1"""
    M0 = view_to_world(p0, w, h)
    M1 = view_to_world(p1, w, h)
    return np.linalg.inv(M1) @ M0

def compute_up_vector(Xs):
    # Ensure the input is a numpy array
    Xs = np.array(Xs)

    # Calculate the matrix as the sum of outer products
    C = np.sum([np.outer(p, p) for p in Xs], axis=0)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Find the eigenvector corresponding to the smallest eigenvalue
    up_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return up_vector

def quad_apparent_AV_rough(p: ImagePair, cam_params):
    h, w = p.img1.shape
    w2, h2 = w / 2, h / 2

    ms = [
        (p.src_pts[:, 0] < w2) & (p.src_pts[:, 1] < h2),
        (p.src_pts[:, 0] > w2) & (p.src_pts[:, 1] < h2),
        (p.src_pts[:, 0] < w2) & (p.src_pts[:, 1] > h2),
        (p.src_pts[:, 0] > w2) & (p.src_pts[:, 1] > h2),
    ]
    vs = [np.linalg.norm(p.src_pts[m] - p.dst_pts[m], axis=1).mean()
          for m in ms if m.any()]

    f = hfov_to_K(cam_params['hfov'], w, h)[1, 1]
    v = np.mean(vs)
    return np.degrees(np.arctan(v / f))

def no_center_apparent_AV_rough(p: ImagePair, cam_params):
    h, w = p.img1.shape
    w2, h2 = w / 2, h / 2
    dists = np.sqrt((p.src_pts[:, 0] - w2) ** 2 + (p.src_pts[:, 1] - h2) ** 2)
    m = dists > h / 3
    f = hfov_to_K(cam_params['hfov'], w, h)[1, 1]
    v = np.linalg.norm(p.src_pts[m] - p.dst_pts[m], axis=1).mean()
    return np.degrees(np.arctan(v / f))

def no_center_apparent_AV(p: ImagePair, cam_params):
    h, w = p.img1.shape
    w2, h2 = w / 2, h / 2
    dists = np.sqrt((p.src_pts[:, 0] - w2) ** 2 + (p.src_pts[:, 1] - h2) ** 2)
    m = dists > h / 3
    f = hfov_to_K(cam_params['hfov'], w, h)[1, 1]
    A, B = p.src_pts.copy(), p.dst_pts.copy()
    A -= np.array([w2, h2])
    B -= np.array([w2, h2])
    A, B = A / f, B / f

    A = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    B = np.concatenate([B, np.ones((B.shape[0], 1))], axis=1)

    A = A / np.linalg.norm(A, axis=1)
    B = B / np.linalg.norm(B, axis=1)

    # Compute arccos angle between vectors
    v = np.sum(A * B, axis=1)
    v = np.clip(v, -1, 1)
    return np.degrees(np.arccos(v)).mean()

def plot_vectors(points):
    # Ensure the input is a numpy array
    points = np.array(points)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin
    origin = np.zeros((points.shape[0], 3))

    # Plot the vectors
    for i in range(points.shape[0]):
        ax.quiver(0, 0, 0, points[i, 0], points[i, 1], points[i, 2], color='b', arrow_length_ratio=0.1)

    # Setting the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Setting the title
    ax.set_title('3D Vectors from Origin')

    # Show the plot
    plt.show()

def alpha_blend(fg_img, bg_img):

    assert fg_img.shape[2] == 4, 'Foreground image must have 4 channels (RGBA)'

    # Ensure the input images are in float format
    fg_img = fg_img.astype(float) / 255
    bg_img = bg_img.astype(float) / 255

    # Extract the alpha channel from the foreground image
    alpha = fg_img[:, :, 3:4]

    # Perform alpha blending
    blended_img = fg_img[:, :, :3] * alpha + bg_img[:, :, :3] * (1 - alpha)

    return (blended_img * 255).astype(np.uint8)

def frame_with_most_loop_closures(pairs):
    unique_indices = np.unique([p.i for p in pairs] + [p.j for p in pairs])
    loop_closures = np.zeros(unique_indices.max() + 1)
    for p in pairs:
        loop_closures[p.i] += 1
        loop_closures[p.j] += 1

    # Find the frame with the most loop closures
    max_idx = np.argmax(loop_closures)

    return max_idx
