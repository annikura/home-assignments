#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))

class Corners:
    MAX_ID = 0

    def __init__(self, coords, sizes, ids=None):
        if ids is None:
            ids = np.arange(self.MAX_ID, self.MAX_ID + sizes.size)
            self.MAX_ID += sizes.size
        self.ids = np.reshape(ids, (-1))
        self.coords = np.reshape(coords, (-1, 2))
        self.sizes = np.reshape(sizes, (-1))
    def filter(self, f):
        bools = []
        for i in range(self.ids.size):
            bools.append(bool(f(self.ids[i], self.coords[i], self.sizes[i])))
        bools = np.array(bools)
        return Corners(self.coords[bools == 1], self.sizes[bools == 1], self.ids[bools == 1])

    def merge(self, corners):
        def f(id, coords, size):
            return not np.any((np.sum((self.coords - coords - 0.0) ** 2, axis=1) <= (size / 2) ** 2))
        corners = corners.filter(f)
        return Corners(np.concatenate((self.coords, corners.coords), axis=0),
                       np.concatenate((self.sizes, corners.sizes), axis=0),
                       np.concatenate((self.ids, corners.ids), axis=0))

    def to_frame_corners(self):
        return FrameCorners(self.ids, self.coords, self.sizes)

    def rescale(self, coef):
        return Corners((self.coords*coef).astype(int), self.sizes*coef, self.ids)

def detect_new_corners(img, feature_params):
    corners = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
    return Corners(corners, np.full(corners.shape[0], feature_params.get("blockSize", 3)))

def detect_new_ranged_corners(img, a, b, step, feature_params):
    result = Corners(np.array([]), np.array([]), np.array([]))
    params = dict(feature_params)

    for scale in [1., 0.75, 0.5]:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        new_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        for i in range(a, b, step):
            if i / scale > 20:
                continue
            params['blockSize'] = i
            params['minDistance'] = i
            result = result.merge(detect_new_corners(new_image, params).rescale(1 / scale))
    return result

def track_corners_lk(old_img, new_img, corners, lk_params):
    old_img = (old_img * 255).astype(np.uint8)
    new_img = (new_img * 255).astype(np.uint8)

    old_coords = np.reshape(corners.coords, (-1, 1, 2)).astype(np.float32)
    new_coords, status, _ = cv2.calcOpticalFlowPyrLK(old_img, new_img, old_coords, None, **lk_params)
    status = np.reshape(status, status.size)

    return Corners(new_coords[status==1], corners.sizes[status==1], corners.ids[status==1])

def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.01,
                          minDistance=3,
                          blockSize=7)
    lk_params = dict(winSize=(5, 5),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.3))

    old_img = frame_sequence[0]
    corners = detect_new_ranged_corners(frame_sequence[0], 5, 15, 3, feature_params)
    builder.set_corners_at_frame(0, corners.to_frame_corners())

    for frame, img in enumerate(frame_sequence[1:], 1):
        corners = track_corners_lk(old_img, img, corners, lk_params)
        corners = corners.merge(detect_new_ranged_corners(img, 5, 15, 3, feature_params))
        builder.set_corners_at_frame(frame, corners.to_frame_corners())
        old_img = img

def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequenc
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
