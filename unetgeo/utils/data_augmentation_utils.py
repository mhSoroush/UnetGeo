from copy import deepcopy

import cv2
import numpy as np


def image_rotation(image, size, boxes, clip_box_coord, angle=10):
    """
    INPUT
        image : numpy array (cv2 image) or None
        size  : (width, height) of image
        boxes : coordinates -> [[x, y], [x, y], ...]
        angle : degree -> 0, 90, 180, 270, ...

    OUTPUT
        img : rotated image
        nboxes : new box coordinates -> [[x, y], [x, y], ...]
    """
    if angle == 0:
        return boxes

    (width, height) = size
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    if image is not None:
        img = np.copy(image)
        img = cv2.warpAffine(img, M, size)
    else:
        img = image

    nboxes = []
    if boxes is not None:
        for box in boxes:
            nbox = []
            for xy in box:
                nxy = M.dot(np.append(xy, 1)).tolist()

                if clip_box_coord:
                    nx, ny = nxy
                    nx = np.clip(nx, 0, width - 1)
                    ny = np.clip(ny, 0, height - 1)
                    nxy = [nx, ny]
                nbox.append(nxy)
            nboxes.append(nbox)

    return img, nboxes

def image_warping(
    image,
    size,
    boxes,
    clip_box_coord,
    n=2.0,
    amp=15.0,
    direction=0,
    normalize_amp=False,
):
    """
    INPUT
        image : numpy array (cv2 image) or None
        size  : (width, height) of image
        n     : the number of sine wave(s)
        amp   : the amplitude of sine wave(s)
        direction : sine wave direction -> 0: vertical  |  1: horizontal
    """

    cols, rows = size
    # todo: use relative amplitude

    if normalize_amp:
        # ave_size = (cols + rows)/2
        ave_size = cols
        move_func = lambda x: amp / 1000 * ave_size * np.sin(2 * np.pi * x / rows * n)
    else:
        move_func = lambda x: amp * np.sin(2 * np.pi * x / rows * n)

    img = image
    if image is not None:
        img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(rows):
            for j in range(cols):
                if direction == 1:
                    offset_x = int(round(move_func(i)))
                    offset_y = 0
                    new_j = j + offset_x
                    if new_j < cols and new_j >= 0:
                        img[i, j] = image[i, new_j % cols]
                    else:
                        img[i, j] = 0

                else:
                    offset_x = 0
                    offset_y = int(round(move_func(j)))
                    new_i = i + offset_y
                    if new_i < rows and new_i >= 0:
                        img[i, j] = image[new_i % rows, j]
                    else:
                        img[i, j] = 0

    nboxes = []
    if boxes is not None:
        for box in boxes:
            nbox = []
            assert len(box) == 4
            for i in range(4):
                x = box[i][0]
                y = box[i][1]
                if direction == 1:
                    nx = x - move_func(y)
                    ny = y
                else:
                    ny = y - move_func(x)
                    nx = x

                if clip_box_coord:
                    nx = np.clip(nx, 0, cols - 1)
                    ny = np.clip(ny, 0, rows - 1)

                nbox.append([nx, ny])
            nboxes.append(nbox)

    return img, nboxes
