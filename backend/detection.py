import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def detect_faces(frame):
    """Returns list of bounding boxes [x, y, w, h]"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    h, w, _ = frame.shape
    boxes = []
    if results.detections:
        for det in results.detections:
            bboxC = det.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            w_box = int(bboxC.width * w)
            h_box = int(bboxC.height * h)
            boxes.append([x, y, w_box, h_box])
    return boxes
