import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

# Paths
VIDEOS_DIR = "video_dataset"
OUTPUT_DIR = "gripper_crops"
GRIPPER_MODEL_PATH = "gripper_detection/yolov8_gripper_model/weights/best.pt"
DESIRED_FPS = 1  # Desired frames per second for sampling
PADDING_FACTOR = 0.3  # 30% padding

# Load YOLOv8 gripper model
gripper_model = YOLO(GRIPPER_MODEL_PATH)

# Make output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get list of video files
video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.avi', '.mkv'))]

# Process each video
for video_file in tqdm(video_files, desc="Processing Videos"):
    video_path = os.path.join(VIDEOS_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]
    save_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(save_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(original_fps // DESIRED_FPS)
    if frame_skip < 1:
        frame_skip = 1

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Detect gripper
        results = gripper_model.predict(source=frame, conf=0.4, verbose=False)
        boxes = results[0].boxes

        if boxes.xyxy is not None and len(boxes.xyxy) > 0:
            for i, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                x1, y1, x2, y2 = map(int, xyxy)

                # Calculate width and height of the box
                box_w, box_h = x2 - x1, y2 - y1

                # Add padding
                pad_w, pad_h = int(box_w * PADDING_FACTOR), int(box_h * PADDING_FACTOR)

                x1_padded = max(x1 - pad_w, 0)
                y1_padded = max(y1 - pad_h, 0)
                x2_padded = min(x2 + pad_w, frame_width - 1)
                y2_padded = min(y2 + pad_h, frame_height - 1)

                # Crop with padding
                crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                crop_filename = f"frame_{saved_count:04d}_{i}.jpg"
                cv2.imwrite(os.path.join(save_dir, crop_filename), crop)
            saved_count += 1

    cap.release()

print("✅ Gripper crops with padding extracted at ~1 FPS!")
