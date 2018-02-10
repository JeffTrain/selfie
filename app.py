import cv2
import dlib
import numpy as np

import os
import os.path

video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()

dir_path = os.path.dirname(os.path.realpath(__file__))
landmark_file_path = os.path.realpath(os.path.join(dir_path, './shape_predictor_68_face_landmarks.dat'))
predictor = dlib.shape_predictor(landmark_file_path)


def rect_to_bounding_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def scale_rect(rect, scale):
    (x, y, w, h) = rect
    return int(x / scale), int(y / scale), int(w / scale), int(h / scale)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def scale_point(point, scale):
    (x, y) = point
    return int(x / scale), int(y / scale)


def trace_face(frame):
    scale = 200 / min(frame.shape[1], frame.shape[0])
    thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    for i, face_rect in enumerate(faces):
        shape = predictor(gray, face_rect)
        shape = shape_to_np(shape)
        for point in shape:
            cv2.circle(frame, scale_point(point, scale), 2, (0, 0, 255), 1)

    return frame


while True:
    ret, frame = video_capture.read()
    face_trace_frame = trace_face(frame)
    cv2.imshow('Video', face_trace_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
