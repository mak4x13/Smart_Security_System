import cv2
import time
import mediapipe as mp

# ==============================
# MediaPipe Face Detection Setup
# ==============================
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detector = mp_face.FaceDetection(
    model_selection=0,          # 0 = short-range (webcam)
    min_detection_confidence=0.6
)

# ==============================
# Camera Init (ONCE)
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

prev_time = 0

print("▶ MediaPipe Face Detection started. Press 'q' to quit.")

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed")
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detector.process(rgb)

    # Draw detections
    if results.detections:
        h, w, _ = frame.shape

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Clamp bounds safely
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("MediaPipe Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
print("✅ Camera released. Exit clean.")

