from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.camera import VideoCamera
from backend.detection import detect_faces
from backend.embedding import get_embedding
from backend.recognition import recognize_face
from backend.enrollment import generate_person_id, add_embedding, finalize_enrollment, CURRENT_SESSION
from backend.attendance import log_attendance, read_attendance
from backend.quality import face_quality
from backend.config import (
    DETECTION_ENHANCE,
    DETECTION_UPSCALE,
    DETECTION_FRAME_SKIP,
    DETECTION_HOLD_FRAMES,
)
from backend.admin import list_persons, delete_person
from backend.pose_validation import analyze_pose_and_draw, validate_expected_pose, POSES
from backend.chatbot import answer_chat

import cv2
import numpy as np
import time
import atexit
import threading
import uuid

app = FastAPI(title="Smart Security System Backend")

SESSION_TTL_SECONDS = 30 * 60
SESSION_MAX_TURNS = 8
SESSION_STORE = {}
SESSION_LOCK = threading.Lock()

# Serve static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/home.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse("frontend/index.html")

@app.get("/chatbot")
def chatbot():
    return FileResponse("frontend/chatbot.html")

# Global Camera Singleton
camera = None
SYSTEM_MODE = "recognition"  # or "enrollment"
OVERLAY_MESSAGE = ""
OVERLAY_COLOR = (0, 0, 255)  # red by default

LAST_BOXES = []
LAST_BOXES_AGE = 0

LAST_FRAME = None

LAST_RECOGNITIONS = []
LAST_RECOGNITION_TIME = 0
STATE_LOCK = threading.Lock()

CAMERA_ACTIVE = True

RECOGNITION_COOLDOWN = 0.5  # seconds
LAST_RECOGNITION_RUN = 0

FACE_TRACKERS = {}  # key: tracker_id, value: tracker object
TRACKER_ID_COUNTER = 0
TRACKER_RECOGNITIONS = {}  # key: tracker_id, value: recognition info

# Ensure camera released on exit
def cleanup():
    release_camera()

atexit.register(cleanup)

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


# MJPEG Video Stream
def gen_frames():
    frame_count = 0
    global LAST_FRAME, CAMERA_ACTIVE, LAST_RECOGNITIONS, LAST_RECOGNITION_TIME, LAST_BOXES, LAST_BOXES_AGE

    while CAMERA_ACTIVE:
        cam = get_camera()
        frame = cam.get_frame()
        if frame is None:
            continue

        with STATE_LOCK:
            LAST_FRAME = frame.copy()

        frame_count += 1

        # Detect faces every N frames; keep last boxes briefly if detection misses.
        if frame_count % DETECTION_FRAME_SKIP == 0:
            detected = detect_faces(
                frame,
                enhance=DETECTION_ENHANCE,
                upscale=DETECTION_UPSCALE
            )
            with STATE_LOCK:
                if detected:
                    LAST_BOXES = detected
                    LAST_BOXES_AGE = 0
                else:
                    LAST_BOXES_AGE += 1
                    if LAST_BOXES_AGE > DETECTION_HOLD_FRAMES:
                        LAST_BOXES = []

        boxes = LAST_BOXES
        current_recognitions = []

        for x, y, w, h in boxes:
            color = (0, 0, 255)
            label = "Unknown"

            if SYSTEM_MODE == "recognition":
                # Safe crop
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                emb = get_embedding(face)
                person, dist = recognize_face(emb)

                if person:
                    label = person["display_name"]
                    color = (0, 255, 0)
                    current_recognitions.append({
                        "person_id": person["person_id"],
                        "display_name": person["display_name"],
                        "role": person["role"],
                        "access_status": person["access_status"],
                        "distance": float(dist) if dist is not None else None
                    })
                else:
                    current_recognitions.append({
                        "person_id": None,
                        "display_name": "Unknown",
                        "role": "",
                        "access_status": "",
                        "distance": None
                    })

            elif SYSTEM_MODE == "enrollment":
                pose_index = CURRENT_SESSION.get("count", 0)
                expected_pose = POSES[min(pose_index, len(POSES) - 1)]

                ok, _, pose_deg, smile_ratio = analyze_pose_and_draw(frame_bgr=frame, draw=True)

                label = f"{expected_pose} ({pose_index + 1}/10)" if pose_index < 10 else "Capture complete"
                color = (255, 255, 0)

                if pose_deg:
                    pitch, yaw, roll = pose_deg
                    cv2.putText(
                        frame,
                        f"pitch={pitch:.1f} yaw={yaw:.1f} roll={roll:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        if current_recognitions:
            with STATE_LOCK:
                LAST_RECOGNITIONS = current_recognitions
                LAST_RECOGNITION_TIME = time.time()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes() +
            b"\r\n"
        )


def expand_box(x, y, w, h, frame_w, frame_h, margin=0.35):
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * (1 + margin), h * (1 + margin)

    x1 = int(max(0, cx - nw / 2))
    y1 = int(max(0, cy - nh / 2))
    x2 = int(min(frame_w, cx + nw / 2))
    y2 = int(min(frame_h, cy + nh / 2))
    return x1, y1, x2, y2


def norm180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@app.post("/camera/stop")
async def camera_stop():
    global CAMERA_ACTIVE

    CAMERA_ACTIVE = False
    release_camera()

    return {"status": "stopped"}

@app.post("/system/mode/{mode}")
async def set_system_mode(mode: str):
    global SYSTEM_MODE, OVERLAY_MESSAGE

    if mode in ["recognition", "enrollment"]:
        SYSTEM_MODE = mode

    # Clear enrollment messages when leaving enrollment
    if SYSTEM_MODE == "recognition":
        OVERLAY_MESSAGE = ""

    return {"mode": SYSTEM_MODE}



@app.get("/video_feed")
def video_feed():
    global CAMERA_ACTIVE

    CAMERA_ACTIVE = True
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Recognition Info Endpoint
@app.get("/recognition/live")
async def recognition_live():
    now = time.time()

    with STATE_LOCK:
        if now - LAST_RECOGNITION_TIME > 3:
            faces = []
        else:
            faces = LAST_RECOGNITIONS.copy()

        for f in faces:
            if f["display_name"] != "Unknown":
                log_attendance(f["person_id"])

    attendance = read_attendance()
    return JSONResponse({
        "faces": faces,
        "attendance": attendance
    })


# Enrollment Start
@app.post("/enroll/start")
async def enroll_start():
    """
    Generate new person_id for enrollment session
    """
    global SYSTEM_MODE
    SYSTEM_MODE = "enrollment"
    pid = generate_person_id()
    return {"person_id": pid}


@app.post("/enroll/capture")
async def enroll_capture():
    """
    Captures one valid sample per call during enrollment:
    - Requires exactly 1 face
    - Requires minimum face quality
    - Calibrates "Look straight" baseline
    - Validates expected pose for current step
    - Requires pose to be valid for 2 consecutive calls
    """
    global OVERLAY_MESSAGE, OVERLAY_COLOR

    with STATE_LOCK:
        frame = None if LAST_FRAME is None else LAST_FRAME.copy()
    if frame is None:
        OVERLAY_MESSAGE = "Camera not ready"
        return {"status": "error", "message": OVERLAY_MESSAGE}

    boxes = detect_faces(
        frame,
        enhance=DETECTION_ENHANCE,
        upscale=DETECTION_UPSCALE
    )
    if len(boxes) != 1:
        OVERLAY_MESSAGE = "Ensure exactly ONE face"
        CURRENT_SESSION["valid_streak"] = 0
        return {"status": "error", "message": OVERLAY_MESSAGE}

    x, y, w, h = boxes[0]

    fh, fw = frame.shape[:2]
    ex1, ey1, ex2, ey2 = expand_box(x, y, w, h, fw, fh, margin=0.45)
    face_for_mesh = frame[ey1:ey2, ex1:ex2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    face = frame[y1:y2, x1:x2]

    if face.size == 0 or face_for_mesh.size == 0:
        OVERLAY_MESSAGE = "Invalid face crop"
        CURRENT_SESSION["valid_streak"] = 0
        return {"status": "error", "message": OVERLAY_MESSAGE}

    face_for_mesh = cv2.resize(face_for_mesh, (320, 320))

    quality = face_quality(face)
    if quality < 2:
        OVERLAY_MESSAGE = "Low Face Quality"
        CURRENT_SESSION["valid_streak"] = 0
        return {"status": "error", "message": OVERLAY_MESSAGE, "quality": quality}

    pose_index = int(CURRENT_SESSION.get("count", 0))
    expected_pose = POSES[min(pose_index, len(POSES) - 1)]

    ok, _, pose_deg, smile_ratio = analyze_pose_and_draw(frame_bgr=face_for_mesh, draw=False)
    if not ok or pose_deg is None:
        return {
            "status": "error",
            "message": "Align face (mesh not detected)",
            "mesh_crop_shape": list(face_for_mesh.shape),
            "det_crop_shape": list(face.shape),
            "quality": quality
        }

    pitch, yaw, roll = pose_deg

    if "pose_center" not in CURRENT_SESSION:
        CURRENT_SESSION["pose_center"] = None
    if "pose_center_samples" not in CURRENT_SESSION:
        CURRENT_SESSION["pose_center_samples"] = []
    if "valid_streak" not in CURRENT_SESSION:
        CURRENT_SESSION["valid_streak"] = 0

    if CURRENT_SESSION["pose_center"] is None and CURRENT_SESSION["count"] == 0:
        CURRENT_SESSION["pose_center_samples"].append((pitch, yaw, roll))

        if len(CURRENT_SESSION["pose_center_samples"]) >= 3:
            arr = np.array(CURRENT_SESSION["pose_center_samples"], dtype=np.float64)
            center = tuple(np.mean(arr, axis=0).tolist())
            CURRENT_SESSION["pose_center"] = center
            CURRENT_SESSION["pose_center_samples"] = []
            CURRENT_SESSION["valid_streak"] = 0

            OVERLAY_MESSAGE = "Baseline set. Look straight to capture."
            return {
                "status": "calibrating",
                "message": OVERLAY_MESSAGE,
                "quality": quality,
                "pose_deg": (pitch, yaw, roll),
                "smile_ratio": float(smile_ratio),
            }

        OVERLAY_MESSAGE = "Hold still - calibrating (look straight)"
        return {
            "status": "calibrating",
            "message": OVERLAY_MESSAGE,
            "quality": quality,
            "pose_deg": (pitch, yaw, roll),
            "smile_ratio": float(smile_ratio),
            "calibrating": True,
            "calib_count": len(CURRENT_SESSION["pose_center_samples"]),
        }

    center = CURRENT_SESSION.get("pose_center")
    if center is not None:
        pitch -= center[0]
        yaw -= center[1]
        roll -= center[2]

    pitch = norm180(pitch)
    yaw = norm180(yaw)
    roll = norm180(roll)

    pose_deg_centered = (float(pitch), float(yaw), float(roll))

    face_area = w * h
    frame_area = frame.shape[0] * frame.shape[1]

    is_valid, msg = validate_expected_pose(
        expected_pose=expected_pose,
        pose_deg=pose_deg_centered,
        smile_ratio=float(smile_ratio),
        face_box_area=face_area,
        frame_area=frame_area
    )

    if not is_valid:
        OVERLAY_MESSAGE = msg
        CURRENT_SESSION["valid_streak"] = 0
        return {
            "status": "error",
            "message": msg,
            "expected_pose": expected_pose,
            "quality": quality,
            "pose_deg": pose_deg_centered,
            "smile_ratio": float(smile_ratio),
        }

    CURRENT_SESSION["valid_streak"] += 1
    if CURRENT_SESSION["valid_streak"] < 2:
        OVERLAY_MESSAGE = "Good - hold it"
        return {
            "status": "error",
            "message": OVERLAY_MESSAGE,
            "expected_pose": expected_pose,
            "quality": quality,
            "pose_deg": pose_deg_centered,
            "smile_ratio": float(smile_ratio),
        }

    CURRENT_SESSION["valid_streak"] = 0

    emb = get_embedding(face)

    person, dist = recognize_face(emb)
    if person and dist is not None and dist < 0.6:
        OVERLAY_MESSAGE = "Person already exists. Restarting."
        generate_person_id()
        CURRENT_SESSION["pose_center"] = None
        CURRENT_SESSION["pose_center_samples"] = []
        CURRENT_SESSION["valid_streak"] = 0
        return {"status": "duplicate", "message": OVERLAY_MESSAGE}

    OVERLAY_MESSAGE = ""
    done, count = add_embedding(emb)

    return {
        "status": "ok",
        "done": done,
        "count": count,
        "quality": quality,
        "expected_pose": expected_pose,
        "pose_deg": pose_deg_centered,
        "smile_ratio": float(smile_ratio),
    }


# Enrollment Confirm
@app.post("/enroll/confirm")
async def enroll_confirm(request: Request):
    """
    Finalize enrollment using embeddings captured during session
    """
    global SYSTEM_MODE
    data = await request.json()

    display_name = data["display_name"]
    role = data["role"]
    department = data.get("department", "")
    access_status = data.get("access_status", "active")

    finalize_enrollment(
        display_name=display_name,
        role=role,
        department=department,
        access_status=access_status
    )
    SYSTEM_MODE = "recognition"  # return to normal mode

    return {"status": "success"}


# Attendance Today
@app.get("/attendance/today")
async def attendance_today():
    """
    Return all logged attendance entries
    """
    return JSONResponse(read_attendance(today_only=True))


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "Message is required."}, status_code=400)

    now = time.time()
    new_session = False
    with SESSION_LOCK:
        sid = request.cookies.get("chat_session_id")
        if sid in SESSION_STORE and now - SESSION_STORE[sid]["updated"] > SESSION_TTL_SECONDS:
            SESSION_STORE.pop(sid, None)
            sid = None
        if not sid:
            sid = uuid.uuid4().hex
            SESSION_STORE[sid] = {"updated": now, "history": [], "memory": {}}
            new_session = True

        session = SESSION_STORE[sid]
        session["updated"] = now
        history = list(session.get("history", []))
        memory = dict(session.get("memory", {}))

    try:
        answer_data = answer_chat(message, memory, history)
    except Exception:
        answer_data = {"answer": "I am having trouble right now. Please try again.", "meta": {}}

    with SESSION_LOCK:
        session = SESSION_STORE.get(sid)
        if session is not None:
            session["updated"] = now
            session["history"].append({"role": "user", "content": message})
            session["history"].append({"role": "assistant", "content": answer_data.get("answer", "")})
            if len(session["history"]) > SESSION_MAX_TURNS * 2:
                session["history"] = session["history"][-SESSION_MAX_TURNS * 2 :]

            meta = answer_data.get("meta") or {}
            if meta.get("matched_person_id"):
                session["memory"]["last_person_id"] = meta["matched_person_id"]
            if meta.get("matched_display_name"):
                session["memory"]["last_person_name"] = meta["matched_display_name"]

    response = JSONResponse(answer_data)
    if new_session:
        response.set_cookie("chat_session_id", sid, httponly=True, samesite="lax")
    return response


@app.post("/chat/clear")
async def chat_clear(request: Request):
    sid = request.cookies.get("chat_session_id")
    if not sid:
        return {"status": "cleared"}
    with SESSION_LOCK:
        SESSION_STORE.pop(sid, None)
    return {"status": "cleared"}


# admin panel endpoints
@app.get("/admin/persons")
async def get_persons():
    persons = list_persons()
    return {
        "count": len(persons),
        "persons": persons
    }

@app.delete("/admin/person/{person_id}")
async def remove_person(person_id: str):
    delete_person(person_id)
    return {"status": "deleted"}
