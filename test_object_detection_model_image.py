from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your trained model (update path)
model = YOLO('object_detection_model/best.pt')

# Load an image (update path)
img_path = '/media/ihub/9CFC-8D2F/Project/gripper_crops/session_10/frame_0020_0.jpg'
img = cv2.imread(img_path)

# Inference
results = model(img)


# Alternatively, plot with matplotlib (convert BGR to RGB)
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
