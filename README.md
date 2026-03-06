# Smart Security System

A real-time AI security platform for face recognition, multi-pose enrollment, and attendance logging.

The system combines computer vision and vector search to register people, recognize them from a live camera stream, and maintain attendance records. It also includes an optional security chatbot for querying enrolled people and attendance data.

## Key Features

- Real-time face detection and recognition from webcam video.
- Multi-pose enrollment workflow (10 guided capture steps) with face-quality checks.
- Pose validation using head-pose estimation (pitch, yaw, roll) and facial cues.
- Attendance logging with deduplication for same-day repeated recognitions.
- Admin management APIs and UI for listing/deleting registered persons.
- Optional chatbot endpoint for natural-language queries over enrollment and attendance data.

## Tech Stack

- Backend: `FastAPI`, `Uvicorn`
- Vision: `OpenCV`, `MediaPipe`, `keras-facenet` (FaceNet), `TensorFlow`
- Vector DB: `ChromaDB`
- Data storage: CSV files (`persons.csv`, `attendance.csv`)
- Frontend: HTML/CSS/JavaScript served as static files

## Project Structure

```text
Smart_Security_System/
  main.py
  requirements.txt
  backend/
    admin.py
    attendance.py
    camera.py
    chatbot.py
    config.py
    detection.py
    embedding.py
    enrollment.py
    pose_validation.py
    recognition.py
    db/
      create_chroma_db.py
  frontend/
    home.html
    index.html
    chatbot.html
    manage_persons.html
    script.js
    style.css
```

## Setup

### 1. Clone and enter project

```bash
git clone https://github.com/mak4x13/Smart_Security_System.git
cd Smart_Security_System
```

`.git` at the end of the URL is optional for GitHub HTTPS cloning; both with and without `.git` work.

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_key_here
```

If you do not use the chatbot, the rest of the system still runs without this key.

## Run the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:

- Home: `http://localhost:8000/`
- Dashboard: `http://localhost:8000/dashboard`
- Chatbot: `http://localhost:8000/chatbot`

## Enrollment and Recognition Flow

1. Switch dashboard mode to **Enrollment**.
2. Capture guided multi-pose samples (10 stages).
3. Confirm enrollment with person metadata.
4. Switch to **Recognition** for live identification.
5. Attendance gets logged automatically for recognized identities.

## API Endpoints (Core)

- `GET /video_feed` - MJPEG live stream
- `GET /recognition/live` - latest recognition + attendance snapshot
- `POST /system/mode/{mode}` - switch `recognition` or `enrollment`
- `POST /enroll/start` - start enrollment session
- `POST /enroll/capture` - capture one validated sample
- `POST /enroll/confirm` - finalize enrollment
- `GET /attendance/today` - today attendance records
- `GET /admin/persons` - list registered persons
- `DELETE /admin/person/{person_id}` - delete person
- `POST /chat` - chatbot query

## Research Context

This project is being used for a research workflow focused on improving recognition robustness through multi-pose enrollment:

- Phase 1: real-participant enrollment and logging experiments.
- Phase 2: BIWI dataset-based pose evaluation and benchmarking.

## Notes

- Camera index defaults to `0` in `backend/camera.py`.
- Runtime data is stored under `backend/db/`.
- Tune thresholds and detection parameters in `backend/config.py` and `backend/pose_validation.py` based on your camera and environment.

## License

This project is released under the MIT License.
