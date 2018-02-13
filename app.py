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

sun_glasses_file = os.path.realpath(os.path.join(dir_path, './images/sunglasses.png'))


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


def add_sun_glasses(target_image, center_x, center_y, width, height=None):
    sun_glasses = cv2.imread(sun_glasses_file)

    if height is None:
        height = int((sun_glasses.shape[0] / sun_glasses.shape[1]) * width)

    sun_glasses = cv2.resize(sun_glasses, (width, height), interpolation=cv2.INTER_AREA)

    # _, mask = cv2.threshold(cv2.cvtColor(sun_glasses, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    # inv_mask = cv2.bitwise_not(mask)

    start_x = int(center_x - width / 2)
    start_y = int(center_y - height / 2)

    # glasses_area = target_image[start_x:start_x + width, start_y:start_y + height]
    # glasses_area_mask = cv2.bitwise_and(glasses_area, glasses_area, mask=inv_mask)
    #
    # print(mask)
    # print(glasses_area)
    # merged = cv2.add(mask, glasses_area_mask)

    target_image[start_y:start_y + height, start_x:start_x + width] = sun_glasses

    return target_image


def trace_face(frame):
    scale = 200 / min(frame.shape[1], frame.shape[0])
    thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    for i, face_rect in enumerate(faces):
        shape = predictor(gray, face_rect)
        shape = shape_to_np(shape)

        center_x = int(shape[27][0] / scale)
        center_y = int(shape[27][1] / scale)

        add_sun_glasses(frame, center_x, center_y, int(abs(shape[17][0] - shape[26][0]) / scale) + 30)
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
