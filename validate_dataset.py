import os
import cv2

DATASET_DIR = "gripper_dataset/train"
OUTPUT_DIR = "debug_annotations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_annotations(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "gripper", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

# Process all images
for image_file in os.listdir(f"{DATASET_DIR}/images"):
    if image_file.endswith(".png"):
        image_path = f"{DATASET_DIR}/images/{image_file}"
        label_path = f"{DATASET_DIR}/labels/{image_file.replace('.png', '.txt')}"
        output_path = f"{OUTPUT_DIR}/{image_file}"
        
        if os.path.exists(label_path):
            draw_annotations(image_path, label_path, output_path)

print("✅ Annotation visualization complete. Check the 'debug_annotations' folder.")
