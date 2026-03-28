"""
Face / eye landmarks, blink ratio, and gaze ratio (dlib).
Place `shape_predictor_68_face_landmarks.dat` next to this file or in the working directory.
"""
from __future__ import annotations

import os
from math import hypot

import cv2
import dlib
import numpy as np

# Resolve predictor: prefer same directory as this module, then CWD
def _default_predictor_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("shape_predictor_68_face_landmarks.dat",):
        p = os.path.join(here, name)
        if os.path.isfile(p):
            return p
        p = os.path.join(os.getcwd(), name)
        if os.path.isfile(p):
            return p
    return os.path.join(here, "shape_predictor_68_face_landmarks.dat")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(_default_predictor_path())


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    if ver_line_length == 0:
        return 0.0
    return hor_line_length / ver_line_length


def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        left_eye.append([facial_landmarks.part(n).x, facial_landmarks.part(n).y])
    for n in range(42, 48):
        right_eye.append([facial_landmarks.part(n).x, facial_landmarks.part(n).y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye


def get_gaze_ratio(eye_points, facial_landmarks, gray, frame_shape):
    """Iris white balance ratio; needs full grayscale frame and frame (H,W) for mask."""
    height, width = frame_shape[:2]
    left_eye_region = np.array(
        [
            (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y),
        ],
        np.int32,
    )
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = int(np.min(left_eye_region[:, 0]))
    max_x = int(np.max(left_eye_region[:, 0]))
    min_y = int(np.min(left_eye_region[:, 1]))
    max_y = int(np.max(left_eye_region[:, 1]))
    if max_y <= min_y or max_x <= min_x:
        return 1.0
    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    h, w = threshold_eye.shape
    if w < 2:
        return 1.0
    left_side_threshold = threshold_eye[0:h, 0 : int(w / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0:h, int(w / 2) : w]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        return 1.0
    if right_side_white == 0:
        return 5.0
    return left_side_white / right_side_white
