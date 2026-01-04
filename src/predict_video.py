"""
Video Prediction Module
Classifies input videos as Real or Fake using trained model
"""
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from face_detection import extract_face

def predict_deepfake_on_video(video_path, model_path, every_n_frames=30):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % every_n_frames == 0:
            face = extract_face(frame)
            if face is None:
                frame_num += 1
                continue

            img = cv2.resize(face, (224, 224))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prob = model.predict(img, verbose=0)[0][0]
            label = "FAKE" if prob > 0.5 else "REAL"
            predictions.append(label)

        frame_num += 1

    cap.release()
    return predictions
