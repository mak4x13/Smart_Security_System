from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from backend.camera import VideoCamera
from backend.detection import detect_faces
from backend.embedding import get_embedding
from backend.recognition import recognize_face
from backend.enrollment import generate_person_id
from backend.enrollment import add_embedding, finalize_enrollment
from backend.attendance import log_attendance, read_attendance
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.quality import face_quality
from backend.admin import list_persons, delete_person

import cv2
import numpy as np
import time
import atexit
import threading

app = FastAPI(title="Smart Security System Backend")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/home.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse("frontend/index.html")

# Global Camera Singleton
camera = None
SYSTEM_MODE = "recognition"  # or "enrollment"
OVERLAY_MESSAGE = ""
OVERLAY_COLOR = (0, 0, 255)  # red by default

LAST_BOXES = []

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


# def compute_iou(box1, box2):
#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2

#     xi1 = max(x1, x2)
#     yi1 = max(y1, y2)
#     xi2 = min(x1 + w1, x2 + w2)
#     yi2 = min(y1 + h1, y2 + h2)
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

#     box1_area = w1 * h1
#     box2_area = w2 * h2
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area else 0


# MJPEG Video Stream
def gen_frames():
    frame_count = 0
    global LAST_FRAME, CAMERA_ACTIVE, LAST_RECOGNITIONS, LAST_RECOGNITION_TIME, LAST_BOXES

    while CAMERA_ACTIVE:
        cam = get_camera()
        frame = cam.get_frame()
        if frame is None:
            continue

        with STATE_LOCK:
            LAST_FRAME = frame.copy()

        frame_count += 1

        # Detect faces every 3rd frame
        if frame_count % 3 == 0:
            with STATE_LOCK:
                LAST_BOXES = detect_faces(frame)

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
                label = "Enrollment Mode"
                color = (255, 255, 0)

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


    # global camera
    # camera = get_camera()
    # cam = get_camera()
    # frame = cam.get_frame()

    # if frame is None:
    #     return JSONResponse({"faces": [], "attendance": []})

    # faces_info = []

    # for _ in LAST_BOXES:
    #     faces_info.append({
    #         "display_name": "Detected",
    #         "role": "",
    #         "access_status": "",
    #         "distance": None
    #     })

    # attendance = read_attendance()
    # return JSONResponse({"faces": faces_info, "attendance": attendance})

    # frame = camera.get_frame()
    # boxes = detect_faces(frame)
    # faces_info = []

    # for box in boxes:
    #     x, y, w, h = box
    #     # Safe cropping
    #     x1, y1 = max(0, x), max(0, y)
    #     x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    #     face_img = frame[y1:y2, x1:x2]

    #     if face_img.size == 0:
    #         continue

    #     emb = get_embedding(face_img)
    #     person, distance = recognize_face(emb)

    #     if person:
    #         faces_info.append({
    #             "display_name": person["display_name"],
    #             "role": person["role"],
    #             "access_status": person["access_status"],
    #             "distance": distance
    #         })
    #         log_attendance(person["person_id"])
    #     else:
    #         faces_info.append({
    #             "display_name": "Unknown",
    #             "role": "",
    #             "access_status": "",
    #             "distance": None
    #         })

    # attendance = read_attendance()

    # return JSONResponse({"faces": faces_info, "attendance": attendance})


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
    global OVERLAY_MESSAGE, OVERLAY_COLOR

    frame = LAST_FRAME
    if frame is None:
        return {"status": "error", "message": "Camera not ready"}

    boxes = detect_faces(frame)
    if len(boxes) != 1:
        OVERLAY_MESSAGE = "Ensure exactly ONE face"
        return {"status": "error"}

    x, y, w, h = boxes[0]
    face = frame[y:y+h, x:x+w]

    quality = face_quality(face)
    if quality < 2:
        OVERLAY_MESSAGE = "Low Face Quality"
        return {"status": "error"}

    emb = get_embedding(face)

    # Duplicate check
    person, dist = recognize_face(emb)
    if person and dist < 0.6:
        OVERLAY_MESSAGE = "Person Already Exists — Restarting"
        generate_person_id()   # restart enrollment
        return {"status": "duplicate"}

    # OK case
    OVERLAY_MESSAGE = ""
    done, count = add_embedding(emb)

    return {
        "status": "ok",
        "done": done,
        "count": count,
        "quality": quality
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