#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    compute_reprojection_errors,
    pose_to_view_mat3x4
)
from _corners import (
    FrameCorners,
)

MIN_STARTING_POINTS = 5
MIN_DIST = 4
MAX_DIST = 60
DIST_STEP = 2

triang_params = TriangulationParameters(max_reprojection_error=1.,
                                        min_triangulation_angle_deg=1.,
                                        min_depth=0.1)


def build_index_intersection(ids1, ids2):
    _, intersection = snp.intersect(ids1.flatten(),
                                    ids2.flatten(),
                                    indices=True)
    return intersection


class FrameTrack:
    MIN_INLIERS = 10

    def __init__(self, id, corners: FrameCorners, mtx=None):
        self.changeble = mtx is None
        self.mtx = mtx
        self.corners = corners
        self.id = id
        self.current_reproject_error = None

    def update_reproj_error(self, cloud: PointCloudBuilder, camera):
        ids1, ids2 = build_index_intersection(cloud.ids, self.corners.ids)
        self.current_reproject_error = compute_reprojection_errors(cloud.points[ids1], self.corners.points[ids2],
                                                                   camera @ self.mtx).mean()

    def pnp(self, cloud: PointCloudBuilder, camera):
        if not self.changeble:
            return []

        ids1, ids2 = build_index_intersection(cloud.ids, self.corners.ids)
        try:
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(cloud.points[ids1],
                                                          self.corners.points[ids2],
                                                          camera,
                                                          None)
        except:
            return []
        if not ret or len(inliers) < FrameTrack.MIN_INLIERS:
            return []
        potential_new_mtx = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        potential_reprojection_error = compute_reprojection_errors(cloud.points[ids1],
                                                                   self.corners.points[ids2],
                                                                   camera @ potential_new_mtx)
        if self.current_reproject_error is None or potential_reprojection_error.mean() < self.current_reproject_error:
            self.mtx = potential_new_mtx
            self.current_reproject_error = potential_reprojection_error.mean()
            # print((potential_reprojection_error[ids1[potential_reprojection_error <  1]] > 1).sum())
            return inliers
        return []

def triangulate_trackers(t1: FrameTrack, t2: FrameTrack, camera, params):
    corrs = build_correspondences(t1.corners, t2.corners)
    points, ids, _ = triangulate_correspondences(corrs,
                                                 t1.mtx,
                                                 t2.mtx,
                                                 camera,
                                                 params)
    return ids, points


def track(iters, trackers, cloud, camera):
    for t in trackers:
        t.pnp(cloud, camera)
    for iter in range(iters):
        best_pairs = []
        for i, t1 in enumerate(trackers):
            for j in range(i + MIN_DIST, min(i + MAX_DIST, len(trackers)), DIST_STEP):
                t2 = trackers[j]
                if t1.mtx is not None and t2.mtx is not None:
                    best_pairs.append((t1, t2))
        best_pairs = sorted(best_pairs,
                            key=lambda x: x[0].current_reproject_error + x[1].current_reproject_error,
                             reverse=True)
        best_pairs = best_pairs[len(best_pairs) // 2:]
        for i, (t1, t2) in enumerate(best_pairs):
            ids, points = triangulate_trackers(t1, t2, camera, triang_params)
            if len(points):
                inliers = np.array(t2.pnp(cloud, camera))
                print(inliers)
                if len(inliers) > 0:
                    _, inline_ids = build_index_intersection(inliers, ids)
                    cloud.add_points(ids[inline_ids], points[inline_ids])
                print("\rIteration {}/{}, triangulating pair {}/{}"
                      .format(iter + 1, iters, i, len(best_pairs)), end=' ' * 20)
        print("\rIteration {}/{}, {} points in the cloud"
              .format(iter + 1, iters, len(cloud.ids)))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    video_size = len(rgb_sequence)
    frame_trackers = [FrameTrack(i, corner_storage[i]) for i in range(video_size)]
    known_tracker_creator = lambda x: FrameTrack(x[0], corner_storage[x[0]], pose_to_view_mat3x4(x[1]))

    frame_trackers[known_view_1[0]] = known_tracker_creator(known_view_1)
    frame_trackers[known_view_2[0]] = known_tracker_creator(known_view_2)
    ids, points = triangulate_trackers(frame_trackers[known_view_1[0]],
                                       frame_trackers[known_view_2[0]],
                                       intrinsic_mat,
                                       triang_params)
    if len(points) < MIN_STARTING_POINTS:
        print(f"Not enough starting points ({len(points)}), please choose another initial frames pair")
        return [], PointCloudBuilder()

    point_cloud_builder = PointCloudBuilder(ids, points)

    frame_trackers[known_view_1[0]].update_reproj_error(point_cloud_builder, intrinsic_mat)
    frame_trackers[known_view_2[0]].update_reproj_error(point_cloud_builder, intrinsic_mat)

    track(4, frame_trackers, point_cloud_builder, intrinsic_mat)

    view_mats = [x.mtx for x in frame_trackers]
    for i in range(1, len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
