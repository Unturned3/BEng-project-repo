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

import common

def main():

    DATA_DIR = '/Users/richard/Desktop/Motorized-Dataset/'
    TRAJ_DIR = DATA_DIR

    in_vid_paths = sorted(glob(DATA_DIR + '*.mp4'))

    for in_vid_path in in_vid_paths[:]:

        in_vid_stem = Path(in_vid_path).stem

        frames = utils.load_video(in_vid_path,
                                grayscale=True)
        #masks = np.load(DATA_DIR + in_vid_stem + '-masks.npz')['masks']
        masks = np.array([np.full_like(frames[0], 0, dtype=np.uint8)] * len(frames))
        video_width = frames[0].shape[1]

        kf_interval = 30

        print('Using video:', in_vid_stem)

        override = True

        if override:
            matcher = ImageMatcher(frames, human_masks=masks, keyframe_interval=kf_interval)
        else:
            matcher = None

        img_pairs_path = DATA_DIR + in_vid_stem + '-pairs.h5'
        img_pairs, closures = common.compute_and_export_image_pairs(
            img_pairs_path, frames, matcher,
            kf_interval, override=override)

        opt_cmd = [
            '../Ceres/build/estimate', DATA_DIR, in_vid_stem + '-pairs.h5',
            'no',  # enable loop closures
            'no',  # load existing camera params
            'no',   # use_stills
        ]
        ret = subprocess.run(opt_cmd)
        if ret.returncode != 0:
            print(f'Ceres exited with non-zero return code!')

        angular_err_abs, hfov_err_abs, cam_params_est = common.compute_errors(
            in_vid_path, video_width, show=False)

        idx2pair, av = common.graph_AV(
            img_pairs, frames, cam_params_est, show=False)

        #still_thresh = float(input('Enter still threshold: '))
        still_thresh = 0.2

        common.export_AV(
            idx2pair, av, still_thresh, img_pairs, img_pairs_path)




if __name__ == '__main__':
    main()
