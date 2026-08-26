# 🚦 Traffic Sign Detection System

A YOLO-based computer vision application for detecting and recognizing traffic signs using a custom-trained object detection model.

## 📌 Project Overview

This project implements a traffic sign detection system using the YOLO object detection framework.

The application provides a simple graphical interface where users can:

- 📷 Start a webcam
- 🛑 Stop the webcam
- 🖼️ Upload an image
- 🔍 Detect traffic signs
- 📦 Display bounding boxes around detected signs
- 📊 Display detection results and confidence information

The system is designed to demonstrate the complete workflow of a custom object detection project, from dataset preparation and model training to real-time inference.

---

## 🧠 Technology Stack

- **Python**
- **YOLO / Ultralytics**
- **OpenCV**
- **Pillow**
- **PyTorch**
- **Computer Vision**
- **Object Detection**

YOLO models are commonly used for real-time object detection because they can perform detection and classification in a single inference pipeline. :contentReference[oaicite:0]{index=0}

---

## 🏗️ Project Structure

```text
yolov8-custom-object-detection/
│
├── test/
│   ├── images/
│   └── labels/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── data.yaml
├── main.py
├── requirements.txt
├── yolov8n.pt
├── yolo26n.pt
├── .gitignore
└── README.md
