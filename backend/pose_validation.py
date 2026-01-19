import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

POSES = [
    "Look straight",
    "Turn left",
    "Turn right",
    "Look up",
    "Look down",
    "Tilt head left",
    "Tilt head right",
    "Smile",
    "Neutral face",
    "Move slightly back",
]

def _norm_angle(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)

def _estimate_head_pose(frame_bgr, face_landmarks):
    h, w, _ = frame_bgr.shape

    idxs = [1, 152, 33, 263, 61, 291]
    pts_2d = []
    for i in idxs:
        lm = face_landmarks.landmark[i]
        pts_2d.append([lm.x * w, lm.y * h])
    pts_2d = np.array(pts_2d, dtype=np.float64)

    pts_3d = np.array([
        [0.0, 0.0, 0.0],        # Nose tip
        [0.0, -63.6, -12.5],    # Chin
        [-43.3, 32.7, -26.0],   # Left eye outer
        [43.3, 32.7, -26.0],    # Right eye outer
        [-28.9, -28.9, -24.1],  # Left mouth corner
        [28.9, -28.9, -24.1],   # Right mouth corner
    ], dtype=np.float64)

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles

    pitch = _norm_angle(pitch)
    yaw = _norm_angle(yaw)
    roll = _norm_angle(roll)

    return float(pitch), float(yaw), float(roll)

def _smile_score(face_landmarks):
    left = face_landmarks.landmark[61]
    right = face_landmarks.landmark[291]
    up = face_landmarks.landmark[13]
    down = face_landmarks.landmark[14]

    mouth_width = np.hypot(right.x - left.x, right.y - left.y)
    mouth_open = np.hypot(down.x - up.x, down.y - up.y) + 1e-6

    return float(mouth_width / mouth_open)

def analyze_pose_and_draw(frame_bgr, draw=True):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return False, None, None, 0.0

    landmarks = res.multi_face_landmarks[0]
    pose_deg = _estimate_head_pose(frame_bgr, landmarks)
    smile_score = _smile_score(landmarks)

    if draw:
        mp_drawing.draw_landmarks(
            image=frame_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=frame_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    return True, landmarks, pose_deg, smile_score

def validate_expected_pose(expected_pose: str, pose_deg, smile_ratio: float, face_box_area: int, frame_area: int):
    if pose_deg is None:
        return False, "Pose estimation failed"

    pitch, yaw, roll = pose_deg

    # Wider straight window for noisy webcams.
    STRAIGHT_YAW = 25
    STRAIGHT_PITCH = 25
    STRAIGHT_ROLL = 25

    TURN_YAW = 22
    UP_PITCH = 15
    DOWN_PITCH = 10
    TILT_ROLL = 12

    face_ratio = face_box_area / max(frame_area, 1)
    MOVE_BACK_MAX_RATIO = 0.10

    ep = expected_pose.lower()

    if ep == "look straight":
        if abs(yaw) < STRAIGHT_YAW and abs(pitch) < STRAIGHT_PITCH and abs(roll) < STRAIGHT_ROLL:
            return True, "OK"
        return False, "Look straight (center head)"

    if ep == "turn left":
        if yaw < -TURN_YAW:
            return True, "OK"
        return False, "Turn LEFT more"

    if ep == "turn right":
        if yaw > TURN_YAW:
            return True, "OK"
        return False, "Turn RIGHT more"

    if ep == "look up":
        if pitch < -UP_PITCH:
            return True, "OK"
        return False, "Look UP more"

    if ep == "look down":
        if pitch > DOWN_PITCH:
            return True, "OK"
        return False, "Look DOWN more"

    if ep == "tilt head left":
        if roll < -TILT_ROLL:
            return True, "OK"
        return False, "Tilt LEFT more"

    if ep == "tilt head right":
        if roll > TILT_ROLL:
            return True, "OK"
        return False, "Tilt RIGHT more"

    # Bigger smile_ratio means more smile-like.
    if ep == "smile":
        if smile_ratio > 4.0:
            return True, "OK"
        return False, "Smile clearly"

    if ep == "neutral face":
        if smile_ratio <= 4.0:
            return True, "OK"
        return False, "Neutral face"

    if ep == "move slightly back":
        if face_ratio < MOVE_BACK_MAX_RATIO:
            return True, "OK"
        return False, "Move slightly back"

    return False, "Unknown expected pose"
