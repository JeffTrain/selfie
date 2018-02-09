import cv2
import dlib

video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()


def rect_to_bounding_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def scale_rect(rect, scale):
    (x, y, w, h) = rect
    return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

def trace_face(frame):
    scale = 200 / min(frame.shape[1], frame.shape[0])
    thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    for i, face_rect in enumerate(faces):
        (x, y, w, h) = scale_rect(rect_to_bounding_box(face_rect), scale)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    return frame

while True:
    ret, frame = video_capture.read()
    face_trace_frame = trace_face(frame)
    cv2.imshow('Video', face_trace_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

