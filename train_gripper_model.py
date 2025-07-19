from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # or yolov8s.pt for a slightly larger model

model.train(
    data='gripper_dataset/dataset.yaml',
    epochs=10,
    imgsz=640,
    batch=16,
    device='cpu',  # Use 0 for GPU, 'cpu' for CPU
    project='gripper_detection',
    name='gripper_detection_model'
)
print("✅ Training complete. The model is saved in the 'gripper_detection/gripper_detection_model' directory.")
