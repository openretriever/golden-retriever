import cv2
import numpy as np


def decode_frames_from_video_path(path: str) -> list:
    frames = []
    cam = cv2.VideoCapture(path)
    not_done = True
    while not_done:
        not_done, frame = cam.read()
        frames.append(frame)
    return frames


def get_equally_spaced_frames(x: list, k: int) -> list:
    x_len = len(x)
    return [x[k] for k in np.floor(np.arange(k) * x_len / k).astype(int)]
