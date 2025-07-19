import cv2
import depthai as dai
from ultralytics import YOLO

# Load the YOLOv8 model (replace with your trained model path)
model = YOLO('gripper_detection/yolov8_gripper_model/weights/best.pt')

# Create pipeline for OAK camera
pipeline = dai.Pipeline()

# Define the OAK camera stream (color camera)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create an output stream for the camera frames
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# Start the pipeline
with dai.Device(pipeline) as device:
    queue_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        # Get frame from OAK camera
        in_rgb = queue_rgb.get()
        frame = in_rgb.getCvFrame()

        # Run YOLOv8 inference
        results = model.predict(source=frame, imgsz=640, conf=0.1, verbose=False)

        # Draw detections on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("Gripper Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
