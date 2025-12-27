from keras_facenet import FaceNet
import numpy as np
import cv2

embedder = FaceNet()

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    emb = embedder.embeddings([face_img])[0]
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb
