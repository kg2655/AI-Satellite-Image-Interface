<div align="center">
  <h1>🌍 AI Satellite Image Interface</h1>
  <p><strong>A state-of-the-art Computer Vision interface for high-resolution Satellite Image Change Detection</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
  [![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
</div>

<br>

## 🚀 Overview

The **AI Satellite Image Interface** is a fully integrated full-stack application designed to automatically detect and highlight topographical changes between two temporal satellite images (Before & After). By deeply analyzing landscapes, infrastructure, and urban spread, this interface helps researchers, educators, and organizations instantly identify planetary spatial changes over time!

Powered by a highly accurate **Siamese U-Net** Deep Learning architecture built in PyTorch, it evaluates high-resolution geospatial imagery and serves the beautiful, colorful results straight to a modern interactive web dashboard.

---

## 🔥 Key Features

- **Advanced Siamese U-Net AI:** Built precisely for satellite change detection leveraging the robust LEVIR-CD dataset.
- **High-Accuracy Pipelines:** Achieving **> 97% Pixel-wise Accuracy** with dedicated training routines and dynamic loss optimizations (Dice + BCE).
- **Interactive Web Interface:** A pristine frontend application where users can seamlessly upload pre and post-event images, view side-by-side comparisons, and instantly see the colored overlay of the predicted changes!
- **In-Depth Evaluation Metrics:** Evaluate models directly from the terminal (Accuracy, Precision, Recall, F1, IoU) complete with beautifully formatted, natively generated Confusion Matrix visualizations.
- **Hardware Optimized:** Features Automatic Mixed Precision (AMP) allowing for blazing fast GPU training out of the box.

---

## 🛠️ Technology Stack

- **Frontend:** React, Vite, TailwindCSS (Responsive, beautiful visualization)
- **Backend:** FastAPI, Uvicorn, Python (Handles rapid REST API handshakes & model inference)
- **Deep Learning:** PyTorch, Torchvision, Seaborn / Matplotlib (Training, prediction, & Metrics)

---

## ⚙️ How to Run Locally

To get the complete app up and running on your local machine, you'll need to spin up both the Vite Frontend and the Python Backend.

### 1. Start the Backend (AI Model & APIs)
Open a terminal and navigate to the project directory, then:
```bash
# Navigate to the backend directory
cd backend

# Activate your virtual environment (Windows)
venv\Scripts\activate

# Install the Python requirements
pip install -r requirements.txt

# Start the FastAPI backend server
python -m uvicorn main:app
```

### 2. Start the Frontend (User Interface)
Open a **new** separate terminal window, remain in the main root directory, and run:
```bash
# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
Navigate to your local browser URL provided in the terminal to interact with the dashboard!

---

## 🧠 Model Training & Evaluation

Want to re-train the Siamese U-Net or test the accuracy? Everything happens inside the `/backend` folder.

**Train the Model:**
```bash
python train_model.py
```
*Note: Configured out of the box with GPU hardware acceleration (`cuda`) support. Evaluates across 50 Epochs by default to achieve maximum golden baseline accuracy!*

**Evaluate the Model:**
```bash
python evaluate_model.py
```
This script runs the validation dataset through your latest weights and prints a highly colorful **Confusion Matrix**, while breaking down your Precision, Recall, F1-Score, and Intersection over Union (IoU) statistics.

---

<div align="center">
  <i>Built to observe the Earth from above.</i>
</div>