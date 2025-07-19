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

## 📁 Dataset

A custom dataset was prepared using videos of the robotic arm interacting with objects. The gripper region was cropped and manually annotated.

📥 **[Click here to download the dataset](https://drive.google.com/file/d/YOUR_DATASET_LINK)**  


After download:
- Unzip the dataset.
- Place the folders (`images`, `labels`) in the appropriate training directory as per YOLO format.

---

## 🚀 Quick Start Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/object-recognition-robotic-arm.git
cd object-recognition-robotic-arm
