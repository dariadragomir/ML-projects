import os
import cv2
import glob
import random
import json
import numpy as np
from PIL import Image
from deepface import DeepFace 

DATA_DIR = '' 

OUTPUT_DIR_TRAIN = ''
OUTPUT_DIR_TEST = ''
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)
def predict_sentiment(frame):

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(rgb_frame, actions=["emotion"], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emotions = result["emotion"]
        return emotions

    except Exception as e:
        print(f"DeepFace analysis error: {e}. Returning default neutral score.")
        return {
            "angry": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 1.0
        }

def process_video(video_path, output_predictions):
    predictions = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return predictions
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"FPS not found for video: {video_path}. Skipping...")
        return predictions

    current_frame = 0
    sec = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame >= sec * fps:
            timestamp = sec  # Timestamp in seconds
            sentiment_scores = predict_sentiment(frame)
            predictions.append({
                "timestamp_sec": timestamp,
                "sentiment": sentiment_scores
            })
            sec += 1
        
        current_frame += 1

    cap.release()
    
    with open(output_predictions, "w") as outfile:
        json.dump(predictions, outfile, indent=2)
    
    return predictions

def split_train_test(video_paths, train_ratio=0.8):
    random.shuffle(video_paths)
    n_train = int(len(video_paths) * train_ratio)
    return video_paths[:n_train], video_paths[n_train:]

def main():
    video_folders = glob.glob(os.path.join(DATA_DIR, "video_*"))
    video_folders.sort() 
    
    video_paths = []
    for folder in video_folders:
        mp4_files = glob.glob(os.path.join(folder, "*.mp4"))
        if mp4_files:
            video_paths.extend(mp4_files) 
        else:
            print(f"No mp4 files found in {folder}")

    
    train_videos, test_videos = split_train_test(video_paths, train_ratio=0.8)
    print(f"Found {len(train_videos)} train and {len(test_videos)} test videos.")
    
    for video_path in train_videos:
        folder_name = os.path.basename(os.path.dirname(video_path))
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(OUTPUT_DIR_TRAIN, f"{folder_name}_{file_name}_predictions.json")

        print(f"Processing TRAIN video: {video_path}")
        process_video(video_path, output_file)
    
    for video_path in test_videos:
        folder_name = os.path.basename(os.path.dirname(video_path))
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(OUTPUT_DIR_TRAIN, f"{folder_name}_{file_name}_predictions.json")

        print(f"Processing TEST video: {video_path}")
        process_video(video_path, output_file)

if __name__ == "__main__":
    main()
