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

import cv2
import numpy as np
import asyncio
import atexit

app = FastAPI(title="Smart Security System Backend")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/home.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse("frontend/index.html")


# -------------------------
# Global Camera Singleton
# -------------------------
# camera = VideoCamera()
camera = None
SYSTEM_MODE = "recognition"  # or "enrollment"
OVERLAY_MESSAGE = ""
OVERLAY_COLOR = (0, 0, 255)  # red by default

LAST_BOXES = []



# Ensure camera released on exit
atexit.register(lambda: camera.release())

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


# -------------------------
# MJPEG Video Stream
# -------------------------
def gen_frames():
    frame_count = 0

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_count += 1

        # Detect faces every 3rd frame
        global LAST_BOXES

        if frame_count % 2 == 0:
            LAST_BOXES = detect_faces(frame)

        boxes = LAST_BOXES

        for x, y, w, h in boxes:
            color = (0, 0, 255)
            label = "Face"

            # -------------------------
            # RECOGNITION MODE
            # -------------------------
            if SYSTEM_MODE == "recognition":
                face = frame[y:y+h, x:x+w]
                emb = get_embedding(face)
                person, dist = recognize_face(emb)

                if person:
                    label = person["display_name"]
                    color = (0, 255, 0)
                    log_attendance(person["person_id"])
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

            # -------------------------
            # ENROLLMENT MODE
            # -------------------------
            elif SYSTEM_MODE == "enrollment":
                label = "Enrollment Mode"
                color = (255, 255, 0)

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

            # if SYSTEM_MODE == "enrollment" and OVERLAY_MESSAGE:
            #     cv2.putText(
            #         frame,
            #         OVERLAY_MESSAGE,
            #         (50, 60),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         1.0,
            #         OVERLAY_COLOR,
            #         3
            #     )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes() +
            b"\r\n"
        )


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
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------
# Recognition Info Endpoint
# -------------------------
@app.get("/recognition/live")
async def recognition_live():
    """
    Return JSON info for all detected faces + attendance snapshot
    """
    cam = get_camera()
    frame = cam.get_frame()

    if frame is None:
        return JSONResponse({"faces": [], "attendance": []})

    boxes = detect_faces(frame)
    faces_info = []

    for box in boxes:
        x, y, w, h = box
        # Safe cropping
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        emb = get_embedding(face_img)
        person, distance = recognize_face(emb)

        if person:
            faces_info.append({
                "display_name": person["display_name"],
                "role": person["role"],
                "access_status": person["access_status"],
                "distance": distance
            })
            log_attendance(person["person_id"])
        else:
            faces_info.append({
                "display_name": "Unknown",
                "role": "",
                "access_status": "",
                "distance": None
            })

    attendance = read_attendance()

    return JSONResponse({"faces": faces_info, "attendance": attendance})

# -------------------------
# Enrollment Start
# -------------------------
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

    frame = camera.get_frame()
    if frame is None:
        OVERLAY_MESSAGE = "Camera not ready"
        return {"status": "error"}

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




# -------------------------
# Enrollment Confirm
# -------------------------
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

# -------------------------
# Attendance Today
# -------------------------
@app.get("/attendance/today")
async def attendance_today():
    """
    Return all logged attendance entries
    """
    return JSONResponse(read_attendance(today_only=True))
