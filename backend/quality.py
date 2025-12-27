import cv2
import numpy as np

def face_quality(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    size = face.shape[0] * face.shape[1]

    score = 0
    if blur > 80: score += 1
    if 70 < brightness < 180: score += 1
    if size > 8000: score += 1

    return score
