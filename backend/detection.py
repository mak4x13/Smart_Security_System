import mediapipe as mp
import cv2
from backend.config import (
    DETECTION_MODEL_SELECTION,
    DETECTION_MIN_CONFIDENCE,
)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detector = mp_face_detection.FaceDetection(
    model_selection=DETECTION_MODEL_SELECTION,
    min_detection_confidence=DETECTION_MIN_CONFIDENCE,
)

def _enhance_for_detection(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_faces(frame, enhance=False, upscale=1.0):
    """Returns list of bounding boxes [x, y, w, h]"""
    src_h, src_w = frame.shape[:2]
    frame_det = frame
    if enhance:
        frame_det = _enhance_for_detection(frame_det)
    if upscale and upscale > 1.0:
        new_w = int(src_w * float(upscale))
        new_h = int(src_h * float(upscale))
        frame_det = cv2.resize(frame_det, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(frame_det, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    boxes = []
    if results.detections:
        for det in results.detections:
            bboxC = det.location_data.relative_bounding_box
            x = int(bboxC.xmin * src_w)
            y = int(bboxC.ymin * src_h)
            w_box = int(bboxC.width * src_w)
            h_box = int(bboxC.height * src_h)
            boxes.append([x, y, w_box, h_box])
    return boxes
