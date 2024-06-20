import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path
from glob import glob
import os

itv = 30

input_vids_dir = '/Users/richard/Desktop/YC/'
input_vid_paths = sorted(glob(os.path.join(input_vids_dir, '*.mp4')))

for vid_n, input_vid_path in enumerate(input_vid_paths, start=1):

    cap = cv2.VideoCapture(input_vid_path)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {input_vid_path} ({vid_n}/{len(input_vid_paths)}). "
          f"n_frames: {n_frames}")

    # Create KCF tracker
    tracker = cv2.TrackerKCF_create()

    yaw_err = []
    pitch_err = []

    for i in tqdm(range(0, n_frames, itv)):

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            tqdm.write(f"Warning: could not read frame {i}\n")
            break;

        if i == 0:
            res = 1024
            frame_small = cv2.resize(frame, (res, res // 2), interpolation=cv2.INTER_AREA)
            bbox = cv2.selectROI(
                "Select ROI", frame_small, showCrosshair=True, printNotice=False
            )
            cv2.destroyAllWindows()
            x_cL, y_cL, w, h = (np.array(bbox) * vid_w / res).astype(np.int32)
            x_cU, y_cU = x_cL + w, y_cL + h

        # Crop frame
        frame = frame[y_cL:y_cU, x_cL:x_cU]

        if i == 0:
            track_bbox = cv2.selectROI(
                "Select Tracking Target", frame, showCrosshair=True, printNotice=False
            )
            cv2.destroyAllWindows()
            tracker.init(frame, track_bbox)

        ret, track_bbox = tracker.update(frame)

        if ret:
            tx, ty, tw, th = (np.array(track_bbox)).astype(np.int32)
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
            yaw_err.append((i, tx))
            pitch_err.append((i, ty))
        else:
            tqdm.write(f"Error: tracking failure on frame {i}\n")
            cv2.imshow(f"Frame {i} Tracking Error", frame)
            cv2.waitKey(0)
            break

        cv2.imshow("Window", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()

    # Fake some extra keyframes to make sure interpolation works
    avg_yaw_err = np.diff(np.array(yaw_err)[:, 1]).mean()
    for i in range(3):
        j = yaw_err[-1][0] + itv
        yaw_err.append((j, yaw_err[-1][1] + avg_yaw_err))

    yaw_err = np.array(yaw_err).T.astype(np.float32)
    pitch_err = np.array(pitch_err).T.astype(np.float32)

    deg_per_px = 360 / vid_w

    yaw_err[1] -= yaw_err[1][0]
    yaw_err[1] *= deg_per_px

    pitch_err[1] -= pitch_err[1][0]
    pitch_err[1] *= deg_per_px

    yaw_error_per_min = yaw_err[1][-1] / ((yaw_err.shape[1] - 1) * itv) * 30 * 60
    pitch_error_per_min = pitch_err[1][-1] / ((pitch_err.shape[1] - 1) * itv) * 30 * 60

    # These are the corrections that should be added to the desired pose at each frame
    yaw_corr = np.array([yaw_err[0], yaw_err[1] * -1])

    # Interpolate these correction keyframes to be applied at every frame
    yaw_corrections = np.interp(
        np.arange(0, n_frames),
        yaw_corr[0],
        yaw_corr[1],
    )

    # Plot yaw error vs frame number (every element of yaw_error is 60 frames)
    plt.plot(
        np.arange(0, n_frames * 2, itv)[: yaw_err.shape[1]],
        yaw_err[1],
        label="Yaw Error",
        marker=".",
    )
    plt.plot(
        np.arange(0, n_frames * 2, itv)[: pitch_err.shape[1]],
        pitch_err[1],
        label="Pitch Error",
        marker=".",
    )

    plt.xlabel("Frame Number")
    plt.ylabel("Error (degrees)")
    plt.legend()
    plt.title(
        "Yaw/Pitch Error per Minute: {:.2f}, {:.2f}".format(
            yaw_error_per_min, pitch_error_per_min
        )
    )
    plt.savefig(input_vid_path[:-4] + "-drift.png")
    plt.show()

    print('Generated yaw corrections for {} frames'.format(len(yaw_corrections)))

    yc_path = input_vid_path[:-4] + "-yc.npy"
    print('Saving yaw corrections to', yc_path)
    np.save(yc_path, yaw_corrections)

    print('\n')

