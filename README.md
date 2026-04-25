# 🌍 AI Satellite Image Interface - Change Detection System

![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-blue?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/AI_Model-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Docker](https://img.shields.io/badge/Deployed_With-Docker-2496ED?style=for-the-badge&logo=docker)

A decoupled, full-stack Artificial Intelligence application designed to detect and highlight geographic, urban, and environmental changes by analyzing dual-temporal satellite imagery. 

### 🚀 Live Demo: [ai-satellite-image-interface.vercel.app](https://ai-satellite-image-interface.vercel.app/)

---

## 📖 Overview
Tracking urban sprawl, deforestation, and natural disaster impacts manually is practically impossible. This system automates the process by taking a **Pre-change (Time 1)** satellite image and a **Post-change (Time 2)** satellite image and running them through a custom **Siamese U-Net** deep learning model. The application outputs a precise, pixel-perfect binary mask highlighting exclusively what has changed between the two timelines.

## 🛠️ Technology Stack
This project operates on a modern microservice architecture:

*   **Frontend (User Interface):** React.js, Vite, TailwindCSS
    *   *Hosted on:* Vercel Serverless
*   **Backend (REST API):** Python, FastAPI, Docker
    *   *Hosted on:* Render Container Service
*   **Deep Learning & AI:** PyTorch, Torchvision, OpenCV, NumPy
    *   *Model Architecture:* Siamese U-Net (Shared-Weight Encoders)
    *   *Loss Function:* Combined BCE (Binary Cross Entropy) + Dice Loss

## 🗄️ Dataset
The model was trained heavily on the **LEVIR-CD Dataset**, a massive open-source building change detection dataset containing ultra-high-resolution Google Earth images. 
*   **Challenge Solved:** Overcoming the "Class Imbalance" problem (where 95% of the land doesn't change) using aggressive Dice Loss penalty functions to force the network to identify rare building construction.

## ⚙️ System Architecture Workflow
1. User uploads a Before and After image to the React frontend.
2. React packages the images into a `FormData` object and dispatches an asynchronous `POST` request.
3. The FastAPI server intercepts the images and translates them into rigid PyTorch `256x256` mathematical tensors.
4. The Siamese U-Net passes the dual timelines through a shared-weight Convolutional encoder, compares the absolute spatial differences at the bottleneck, and upscales them into a binary matrix.
5. The matrix is converted into a `base64` image string and relayed back to the user's screen in milliseconds.

## 💻 Local Setup Instructions

If you want to run this application on your local machine:

### 1. Start the Backend (FastAPI)
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Or `source venv/bin/activate` on Mac/Linux
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

### 2. Start the Frontend (React)
Open a new terminal window:
```bash
# Ensure you are in the root directory
npm install
npm run dev
```

*Note: You must have `siamese_unet_full.pth` (the model weights) inside the `/backend` folder for the server to successfully boot.*

---
*Built as a comprehensive research and development capstone project.*