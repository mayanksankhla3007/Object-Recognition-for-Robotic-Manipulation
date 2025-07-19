# train_gripper_object_model.py

from ultralytics import YOLO

# Path to dataset.yaml (update if needed)
DATASET_YAML = 'gripper_crops_dataset/dataset.yaml'  # <-- Update this!

# Load YOLOv8m pretrained model
model = YOLO('yolov8m.pt')

# Train the model
model.train(
    data=DATASET_YAML,
    epochs=30,               # Moderate epochs to stay within time limit
    imgsz=640,               # Standard image size
    batch=8,                 # Lower batch size for CPU
    device=0,          
    workers=2,               # Disable multiprocessing for better CPU stability
)

# After training, the model will be saved in:
# object_detection_gripper/yolov8m_object_model/weights/best.pt

