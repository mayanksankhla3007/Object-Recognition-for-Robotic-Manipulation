# 🤖 Object Recognition for Robotic Manipulation and Gripper-Based Interaction

This project enables a robotic arm to **recognize and classify the object** it is manipulating in **real time** using a two-stage YOLOv8-based deep learning pipeline. It uses the **Luxonis OAK-D camera** to capture video, detects the **gripper**, then performs **object recognition** within the gripper crop using a second YOLOv8 model.

---

## 📌 Project Features

- Real-time object detection using two custom-trained YOLOv8 models.
- Fine-tuned on a custom dataset of gripper-object interactions.
- Uses Luxonis OAK-D camera for live RGB video input.
- Modular training and inference pipeline.
- Fully documented dataset preparation and model training flow.


---

## 📁 Dataset & Pretrained Models

📥 **[Download Dataset + Trained Models](https://drive.google.com/drive/folders/1I_5nQUgrvlu9U8h595vCt1r-NStuat5S)**

This shared folder contains:
- `gripper_detection_model.pt` – model for gripper detection
- `object_recognition_model.pt` – model for object recognition
- `gripper_dataset/` – Annotated YOLO-format dataset for gripper detection
- `object_recognition_dataset/` – Annotated YOLO-format dataset for object recognition


---

## 🚀 Quick Start Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mayanksankhla3007/Object-Recognition-for-Robotic-Manipulation.git
cd object-recognition-robotic-arm
