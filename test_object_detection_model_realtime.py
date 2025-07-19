
import cv2
import numpy as np
from ultralytics import YOLO
import depthai as dai

def draw_label(img, label, x, y, color_box):
    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    img_h, img_w = img.shape[:2]

    # Try drawing label above the box
    label_y = y - 10

    # If label goes out of top boundary, draw below box
    if label_y - h - baseline < 0:
        label_y = y + h + 10
        if label_y + baseline > img_h:
            label_y = img_h - baseline - 1

    # Clamp x inside image boundaries
    label_x = max(0, min(x, img_w - w))

    # Draw filled rectangle for label background
    cv2.rectangle(img,
                  (label_x, label_y - h - baseline),
                  (label_x + w, label_y + baseline),
                  color_box,
                  thickness=cv2.FILLED)
    # Draw label text
    cv2.putText(img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

def main():
    # Load your trained YOLO models
    gripper_model = YOLO('gripper_detection_model.pt')
    object_model = YOLO('object_recognition_model.pt')

    # Setup DepthAI pipeline
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    device = dai.Device(pipeline)
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    pad = 20  # padding for gripper bbox expansion

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is None:
            continue
        frame = in_rgb.getCvFrame()

        # --- Gripper detection ---
        gripper_results = gripper_model(frame)[0]

        # Draw gripper and detect objects inside
        for det in gripper_results.boxes:
            conf = det.conf.cpu().item()
            
            bbox = det.xyxy.cpu().numpy().flatten()  # flatten to 1D array of 4 values
            x1, y1, x2, y2 = bbox.astype(int)


            # Expand bbox with padding and clamp to frame size
            x1_exp = max(0, x1 - pad)
            y1_exp = max(0, y1 - pad)
            x2_exp = min(frame.shape[1], x2 + pad)
            y2_exp = min(frame.shape[0], y2 + pad)

            # Crop expanded bbox region for object detection
            gripper_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

            # Run object detection on gripper crop
            object_results = object_model(gripper_crop)[0]

            # Draw expanded gripper bbox on original frame (blue)
            cv2.rectangle(frame, (x1_exp, y1_exp), (x2_exp, y2_exp), (255, 0, 0), 2)

            # Draw gripper label above expanded bbox
            cls_id = int(det.cls.cpu().numpy())
            label = f'{gripper_model.names[cls_id]} {conf:.2f}'
            draw_label(frame, label, x1_exp, y1_exp - 10, (255, 0, 0))

            # Draw object detections inside gripper crop
            for obj_det in object_results.boxes:
                obj_conf = obj_det.conf.cpu().item()
                
                obj_bbox = obj_det.xyxy.cpu().numpy().flatten()
                ox1, oy1, ox2, oy2 = obj_bbox.astype(int)

                obj_cls_id = int(obj_det.cls.cpu().numpy())
                class_name = object_model.names[obj_cls_id]
                obj_label = f'{class_name} {obj_conf:.2f}'

                # Draw bbox on original frame but adjusted to original frame coords
                # Because object bbox coords are relative to gripper_crop,
                # add offset (x1_exp, y1_exp) to map back to frame coords
                frame_ox1, frame_oy1 = ox1 + x1_exp, oy1 + y1_exp
                frame_ox2, frame_oy2 = ox2 + x1_exp, oy2 + y1_exp

                cv2.rectangle(frame, (frame_ox1, frame_oy1), (frame_ox2, frame_oy2), (0, 255, 0), 2)
                draw_label(frame, obj_label, frame_ox1, frame_oy1 - 10, (0, 255, 0))

        cv2.imshow("Gripper + Object Detection", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
