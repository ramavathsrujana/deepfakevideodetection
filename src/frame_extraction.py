"""
Frame Extraction Module
Extracts frames from input videos at fixed intervals
"""
import cv2
import os

def extract_all_videos(input_folder, output_folder, label, every_n_frames=30):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

    for video in video_files:
        video_path = os.path.join(input_folder, video)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % every_n_frames == 0:
                frame_path = os.path.join(output_folder, f"{label}_{total_saved}.jpg")
                cv2.imwrite(frame_path, frame)
                total_saved += 1

            frame_count += 1

        cap.release()
