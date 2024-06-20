import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path
from glob import glob
import os

itv = 30

input_vids_dir = '/Users/richard/Desktop/Raw/'
yc_paths = sorted(glob(os.path.join(input_vids_dir, '*-yc.npy')))

for yc_path in yc_paths:

    yc = np.load(yc_path) * -1

    yaw_error_per_min = np.diff(yc).mean() * 30 * 60

    plt.scatter(
        np.arange(0, len(yc)),
        yc,
        label="Yaw Error",
        marker='.',
        s=1,
    )

    plt.xlabel("Frame Number")
    plt.ylabel("Error (degrees)")
    plt.legend()
    plt.title(
        "Average Yaw Error per Minute: {:.2f}".format(
            yaw_error_per_min
        )
    )
    plt.savefig(yc_path[:-4] + '.png')
    plt.close()

