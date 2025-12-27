import cv2
from threading import Thread, Lock
import time

class VideoCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.lock = Lock()
        self.running = True
        self.frame = None

        self.thread = Thread(target=self.update_frame, daemon=True)
        self.thread.start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
