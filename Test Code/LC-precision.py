import numpy as np
import utils
from ImageMatcher import ImageMatcher, ImagePair
import h5py
from scipy.spatial.transform import Rotation as Rot
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from glob import glob
import os
import sys
import copy
from timer import Timer

def main():

    DATA_DIR = '/Users/richard/Desktop/Motorized-Dataset/'
    TRAJ_DIR = DATA_DIR

    in_vid_paths = sorted(glob(DATA_DIR + '*.mp4'))

    for in_vid_path in in_vid_paths[0:5]:

        in_vid_stem = Path(in_vid_path).stem

        frames = utils.load_video(in_vid_path,
                                grayscale=True)
        masks = np.load(DATA_DIR + in_vid_stem + '-masks.npz')['masks']
        video_width = frames[0].shape[1]

        kf_interval = 30

        print('Using video:', in_vid_stem)

        override = False

        if override:
            matcher = ImageMatcher(frames, human_masks=masks, keyframe_interval=kf_interval)
        else:
            matcher = None

        img_pairs_path = DATA_DIR + in_vid_stem + '-pairs.h5'
        image_pairs, closures = compute_and_export_image_pairs(
            img_pairs_path, frames, matcher,
            kf_interval, override=override)

        idx2pair = {}
        for p in image_pairs:
            idx2pair[(p.i, p.j)] = p

        _, cpg = utils.load_est_gt_poses(
        in_vid_path, video_width)

        true_pos = 0
        false_pos = 0
        false_neg = 0

        for i in range(0, len(frames) - kf_interval, kf_interval):
            for j in range(i + kf_interval, len(frames), kf_interval):
                R = Rot.from_matrix(cpg[i]['R'] @ cpg[j]['R'].T)
                angle = np.degrees(R.magnitude())
                if (i, j) in idx2pair:
                    if angle < 70:
                        true_pos += 1
                    else:
                        false_pos += 1
                        print(f'FALSE POS angle: {angle}')
                        utils.visualize_matches(idx2pair[(i, j)])
                else:
                    # Compute proprtion of area of mask[i] that is 1
                    propi = np.sum(masks[i]) / masks[i].size
                    propj = np.sum(masks[j]) / masks[j].size
                    if angle < 50 and propi < 0.3 and propj < 0.3:
                        #print("FALSE NEG:", angle, propi, propj)
                        ## Show frame i j side by side plt
                        #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        #ax[0].imshow(frames[i], cmap='gray')
                        #ax[1].imshow(frames[j], cmap='gray')
                        #plt.show()
                        false_neg += 1
        print("True Positives: ", true_pos)
        print("False Positives: ", false_pos)
        print("False Negatives: ", false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        # Save them to file.
        with open(f'{in_vid_path[:-4]}_LC_PR.txt', 'w') as f:
            f.write(f'Precision: {precision}\n')
            f.write(f'Recall: {recall}\n')

def compute_and_export_image_pairs(img_pairs_path, frames,
                                   matcher, kf_interval, override):
    image_pairs = []
    closures = []
    still_cnt = 0

    if not override and os.path.isfile(img_pairs_path):
        print('Loading existing file', img_pairs_path)
        image_pairs = utils.import_image_pairs(img_pairs_path, frames)
        for p in image_pairs:
            if p.still:
                still_cnt += 1
            if p.i != p.j - 1:
                closures.append(p)
    else:

        with Timer(msg_prefix='Adj frame time: '):
            for i in range(0, len(frames) - 1):
                p = matcher.match(
                    i, i+1,
                    min_match_count=60,
                    ransac_max_iters=2000,
                    #keep_percent=0.7,
                    verbose=True,
                )
                if p is None:
                    print('No match found for frames', i, i+1)
                image_pairs.append(p)

        for i in range(0, len(frames) - kf_interval, kf_interval):
            for j in range(i + kf_interval, len(frames), kf_interval):
                if matcher.sift_kds[i] is None or matcher.sift_kds[j] is None:
                    continue
                p = matcher.match(i, j, 'sift',
                                min_match_count=25)
                if p is None:
                    continue
                closures.append(p)
                image_pairs.append(p)

    print('Number of loop closures:', len(closures))
    print('Total number of image pairs:', len(image_pairs))
    print('Number of still pairs:', still_cnt)

    if override or (not os.path.isfile(img_pairs_path)):
        utils.export_image_pairs(img_pairs_path, image_pairs)

    return image_pairs, closures

def compute_errors(in_vid_path, video_width, save=False):

    cpe, cpg = utils.load_est_gt_poses(
        in_vid_path, video_width)

    cam_params_est, cam_params_gt = utils.set_ref_cam(0, cpe, cpg)

    # Compute the angular pose error and hfov error between cam_params and cam_params_gt
    angular_err_abs = []
    angular_err_rel = []
    hfov_err_abs = []
    hfov_err_rel = []
    cam_indices = []

    for i in sorted(cam_params_est.keys()):
        R = cam_params_est[i]['R']
        R_gt = cam_params_gt[i]['R']

        ae = np.degrees(Rot.from_matrix(R @ R_gt.T).magnitude())
        angular_err_abs.append(ae)
        angular_err_rel.append(ae / max(np.degrees(Rot.from_matrix(R_gt.T).magnitude()), 0.1))

        hfove = np.abs(cam_params_est[i]['hfov'] - cam_params_gt[i]['hfov'])
        hfov_err_abs.append(hfove)
        hfov_err_rel.append(hfove / cam_params_gt[i]['hfov'])
        cam_indices.append(i)
    #angular_err_abs = np.array(angular_err_abs) * 1.5
    print(f'angular error mean: {np.mean(angular_err_abs):.5f}, '
        f'max: {np.max(angular_err_abs):.5f}')
    print(f'hfov error mean: {np.mean(hfov_err_abs):.5f}, max: {np.max(hfov_err_abs):.5f}')

    fig, ax = plt.subplots(dpi=150, figsize=(5, 4))
    ax.set_title('Absolute Error')
    ax.plot(angular_err_abs, label='angular pose')
    ax.plot(hfov_err_abs, label='fov')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Degrees')
    #ax.set_ylim(-0.3, 6.2)
    #ax.grid()
    #ax.set_yticks(np.arange(0, 8.5, 2))
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(f'{in_vid_path[:-4]}_err.png')
        with open(f'{in_vid_path[:-4]}_err.txt', 'w') as f:
            f.write(f'angular error mean: {np.mean(angular_err_abs):.5f}, '
                    f'max: {np.max(angular_err_abs):.5f}\n')
            f.write(f'hfov error mean: {np.mean(hfov_err_abs):.5f}, '
                    f'max: {np.max(hfov_err_abs):.5f}\n')
    plt.show()
    plt.close()

    return angular_err_abs, hfov_err_abs, cam_params_est

def graph_AV(image_pairs, frames, cam_params_est):
    idx2pair = {}
    for p in image_pairs:
        idx2pair[(p.i, p.j)] = p

    av = []
    for i in range(0, len(frames) - 1):
        p = idx2pair[(i, i+1)]
        av.append(utils.avg_quad_angular_velocity(p, cam_params_est[i]))

    # Moving average filter to smooth out the angular velocity
    # preserve the length of the array (padding = same as edge)
    av.insert(0, av[0])
    av.append(av[-1])
    av = np.convolve(av, np.ones(3) / 3, mode='valid')

    still_thresh = 0.2
    fig, ax = plt.subplots(dpi=150, figsize=(5, 4))
    ax.set_title('Apparent Angular Velocity')
    # plot av with thin line width.
    ax.plot(av, color='black', linewidth=0.5)
    ax.scatter(np.arange(0, len(av)), av, c=['red' if a < still_thresh
                                            else 'blue' for a in av], s=5)
    #ax.set_xlim(390, 660)
    #ax.set_ylim(-0.05, 0.65)
    ax.grid()
    ax.set_xlabel('Frame')
    ax.set_ylabel('Degrees per frame')
    fig.tight_layout()
    plt.show()

    return idx2pair, av

def export_AV(idx2pair, av, still_thresh, image_pairs, img_pairs_path):
    cnt = 0
    for i in range(0, len(av)):
        p = idx2pair[(i, i+1)]
        p.still = av[i] < still_thresh
        if p.still:
            cnt += 1
    print(f'Number of still pairs: {cnt}')
    utils.export_image_pairs(img_pairs_path, image_pairs)


if __name__ == '__main__':
    main()
