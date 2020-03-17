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
    eye3x4,
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
    filter_frame_corners,
)

INLINER_FREQUENCY_TRASHHOLD = 0.9
MIN_INLINER_FRAMES = 5
MIN_STARTING_POINTS = 5
ITERATIONS = 5
MAX_REPROJECTION_ERROR = 8
MAX_TRANSLATION = 3

FRAMES_STEP = 10
FRAMES_MIN_WINDOW = 5
FRAMES_MAX_WINDOW = 100

triang_params = TriangulationParameters(max_reprojection_error=MAX_REPROJECTION_ERROR,
                                        min_triangulation_angle_deg=1.,
                                        min_depth=0.001)


def build_index_intersection(ids1, ids2):
    _, intersection = snp.intersect(ids1.flatten(),
                                    ids2.flatten(),
                                    indices=True)
    return intersection


def build_index_difference(ids1, ids2):
    _, intersection = snp.intersect(ids1.flatten(),
                                    ids2.flatten(),
                                    indices=True)
    mask = np.ones_like(ids1).astype(bool)
    mask[intersection[0]] = False
    return np.arange(ids1.size)[mask.flatten()]


class FrameTrack:
    MIN_INLIERS = 2

    def __init__(self, id, corners: FrameCorners, mtx=None):
        self.changeble = mtx is None
        self.mtx = mtx
        self.corners = corners
        self.id = id
        self.current_reproject_error = None
        self.last_inliners = []

    def update_reproj_error(self, cloud: PointCloudBuilder, camera):
        ids1, ids2 = build_index_intersection(cloud.ids, self.corners.ids)
        self.current_reproject_error = compute_reprojection_errors(cloud.points[ids1], self.corners.points[ids2],
                                                                   camera @ self.mtx).mean()

    def pnp(self, cloud: PointCloudBuilder, camera):
        ids1, ids2 = build_index_intersection(cloud.ids, self.corners.ids)
        try:
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(cloud.points[ids1],
                                                          self.corners.points[ids2],
                                                          camera,
                                                          None, reprojectionError=MAX_REPROJECTION_ERROR)
        except:
            return self.last_inliners
        if not ret or len(inliers) < FrameTrack.MIN_INLIERS or np.linalg.norm(tvec) > MAX_TRANSLATION:
            return self.last_inliners
        potential_new_mtx = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        potential_reprojection_error = compute_reprojection_errors(cloud.points[ids1],
                                                                   self.corners.points[ids2],
                                                                   camera @ potential_new_mtx)
        if self.current_reproject_error is None or potential_reprojection_error.mean() < self.current_reproject_error:
            self.mtx = potential_new_mtx
            self.current_reproject_error = potential_reprojection_error.mean()
            self.last_inliners = cloud.ids[ids1][np.array(inliers)]
        return self.last_inliners


def triangulate_trackers(t1: FrameTrack, t2: FrameTrack, camera, params):
    corrs = build_correspondences(t1.corners, t2.corners)
    points, ids, _ = triangulate_correspondences(corrs,
                                                 t1.mtx,
                                                 t2.mtx,
                                                 camera,
                                                 params)
    return ids, points


class FrequencyCounter:
    def __init__(self):
        self.freqs = {}
        self.totals = {}
        self.points = {}

    def add(self, xs_subset, points, xs_total):
        xs_subset = xs_subset.flatten().tolist()
        points = list(points)
        xs_total = xs_total.flatten().tolist()
        for x, p in zip(xs_subset, points):
            self.freqs[x] = self.freqs.get(x, 0) + 1
            self.points[x] = p
        for x in xs_total:
            self.totals[x] = self.totals.get(x, 0) + 1
            self.freqs[x] = self.freqs.get(x, 0)

    def get_freq(self, x):
        return self.freqs[x] / self.totals[x]

    def get_top_x_percent_freq(self, x):
        if not len(self.totals):
            return 0
        return self.get_freq(sorted(self.totals.keys(), key=self.get_freq, reverse=True)[int(x * len(self.totals))])

    def get_freqs_above(self, th, min_in):
        good_ids = sorted([x for x in self.points.keys() if self.get_freq(x) > th and self.freqs[x] > min_in])
        return np.array(
            good_ids).flatten().astype(int), \
               np.array(
                   [self.points[x] for x in good_ids]),


def track(iters, trackers, cloud, camera):
    start_ids, start_points = cloud.ids, cloud.points

    for iter in range(iters):
        for frame_num, t1 in enumerate(trackers):
            print("\rIteration {}/{}, triangulate for frame {}/{}"
                  .format(iter + 1, iters, t1.id, len(trackers)), end=' ' * 20)
            if t1.mtx is None:
                continue
            for j in range(frame_num - FRAMES_MAX_WINDOW, frame_num + FRAMES_MAX_WINDOW + 1, FRAMES_STEP):
                if j < 0 or j >= len(trackers):
                    continue
                t2 = trackers[j]
                if abs(t1.id - t2.id) <= FRAMES_MIN_WINDOW or t2.mtx is None:
                    continue
                corrs = build_correspondences(t1.corners, t2.corners)
                if not len(corrs.ids):
                    continue
                points, ids, _ = triangulate_correspondences(corrs,
                                                             t1.mtx,
                                                             t2.mtx,
                                                             camera,
                                                             triang_params)
                if len(points):
                    cloud.add_points(ids, points)
        fc = FrequencyCounter()
        for t in trackers:
            inliers = t.pnp(cloud, camera)
            inliers = np.array(inliers).flatten().astype(int)
            print("\rIteration {}/{}, PnP for frame {}/{}, {} inliners"
                  .format(iter + 1, iters, t.id, len(trackers), len(inliers)), end=' ' * 20)
            if len(inliers):
                ids1, ids2 = build_index_intersection(cloud.ids, inliers)
                fc.add(inliers[ids2], cloud.points[ids1], t.corners.ids)
        good_points_ids, good_points = fc.get_freqs_above(INLINER_FREQUENCY_TRASHHOLD, MIN_INLINER_FRAMES)
        cloud = PointCloudBuilder(good_points_ids, good_points)
        for t in trackers:
            t.pnp(cloud, camera)

        print("\rIteration {}/{}, {} points in the cloud"
              .format(iter + 1, iters, len(cloud.ids)), end=' ' * 20 + "\n")

    fc = FrequencyCounter()
    for t in trackers:
        inliers = t.pnp(cloud, camera)
        inliers = np.array(inliers).flatten().astype(int)
        if len(inliers):
            ids1, ids2 = build_index_intersection(cloud.ids, inliers)
            fc.add(inliers[ids2], cloud.points[ids1], t.corners.ids)
    good_points_ids, good_points = fc.get_freqs_above(INLINER_FREQUENCY_TRASHHOLD, MIN_INLINER_FRAMES)
    cloud = PointCloudBuilder(good_points_ids, good_points)

    return cloud


def get_best_intersected(corner_storage, min_dist=10, max_dist=50, min_intersections=10, max_frame=200):
    assert min_dist > 0
    variants = []
    for i, corners1 in enumerate(corner_storage):
        for j in range(i + min_dist, i + max_dist):
            if j >= len(corner_storage) or j >= max_frame:
                break
            ids, _ = build_index_intersection(corner_storage[i].ids, corner_storage[j].ids)
            if len(ids) > min_intersections:
                variants.append((i, j, len(ids)))
    return sorted(variants, key=lambda x: (x[2] // 20, x[1] - x[0]), reverse=True)


def get_matrix_poses(corner_storage, intrisinc_mat):
    pairs = get_best_intersected(corner_storage)
    best_pair = -1, -1
    best_pair_result = -1

    for i, j, _ in pairs[:100]:
        ids1, ids2 = build_index_intersection(corner_storage[i].ids, corner_storage[j].ids)
        points1 = corner_storage[i].points[ids1]
        points2 = corner_storage[j].points[ids2]

        E, mask = cv2.findEssentialMat(points1, points2, focal=intrisinc_mat[0][0])
        if mask.sum() < 10:
            continue
        F, mask = cv2.findFundamentalMat(points1, points2)
        if mask.sum() < 10:
            continue
        _, R, t, mask = cv2.recoverPose(E, points1, points1, focal=intrisinc_mat[0][0])
        if mask.sum() < 10:
            continue

        corrs = build_correspondences(corner_storage[i], corner_storage[j])
        points, ids, _ = triangulate_correspondences(corrs,
                                                     eye3x4(),
                                                     np.hstack((R, t)),
                                                     intrisinc_mat, TriangulationParameters(
                                                         max_reprojection_error=5,
                                                         min_triangulation_angle_deg=5.,
                                                         min_depth=0.001))
        current_result = len(ids) // 20
        if current_result > best_pair_result:
            best_pair = i, j
            best_pair_result = current_result

    i, j = best_pair

    ids1, ids2 = build_index_intersection(corner_storage[i].ids, corner_storage[j].ids)
    points1 = corner_storage[i].points[ids1]
    points2 = corner_storage[j].points[ids2]

    E, mask = cv2.findEssentialMat(points1, points2, focal=intrisinc_mat[0][0])
    F, mask = cv2.findFundamentalMat(points1, points2)
    _, R, t, mask = cv2.recoverPose(E, points1, points1, focal=intrisinc_mat[0][0])

    print(f"Chosen frames {i} and {j}")

    return (i, view_mat3x4_to_pose(eye3x4())),\
           (j, view_mat3x4_to_pose(np.hstack((R, t))))


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = get_matrix_poses(corner_storage, intrinsic_mat)

    video_size = len(rgb_sequence)
    frame_trackers = [FrameTrack(i, corner_storage[i]) for i in range(video_size)]
    known_tracker_creator = lambda x: FrameTrack(x[0], corner_storage[x[0]], pose_to_view_mat3x4(x[1]))

    frame_trackers[known_view_1[0]] = known_tracker_creator(known_view_1)
    frame_trackers[known_view_2[0]] = known_tracker_creator(known_view_2)

    init_params = triang_params

    for angle in range(90, 0, -2):
        params = TriangulationParameters(max_reprojection_error=MAX_REPROJECTION_ERROR,
                                         min_triangulation_angle_deg=angle,
                                         min_depth=0.001)
        _, points = triangulate_trackers(frame_trackers[known_view_1[0]],
                                         frame_trackers[known_view_2[0]],
                                         intrinsic_mat,
                                         params)
        if len(points) > 100:
            print(f"Chosen init angle: {angle}")
            init_params = params
            break

    ids, points = triangulate_trackers(frame_trackers[known_view_1[0]],
                                       frame_trackers[known_view_2[0]],
                                       intrinsic_mat,
                                       init_params)

    point_cloud_builder = PointCloudBuilder(ids, points)
    if len(points) < MIN_STARTING_POINTS:
        print(f"Not enough starting points ({len(points)}), please choose another initial frames pair"
              f"\n0, 20 is a good pair for short fox, ")
        return [], PointCloudBuilder()

    frame_trackers[known_view_1[0]].update_reproj_error(point_cloud_builder, intrinsic_mat)
    frame_trackers[known_view_2[0]].update_reproj_error(point_cloud_builder, intrinsic_mat)

    point_cloud_builder = track(ITERATIONS, frame_trackers, point_cloud_builder, intrinsic_mat)

    view_mats = [x.mtx for x in frame_trackers]
    for i in range(1, len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]
    for i in range(len(view_mats) - 2, -1, -1):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i + 1]

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