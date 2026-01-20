# Human Facial Emotion Recognition

This repository contains a complete **Facial Emotion Recognition** system using deep learning, served with a Flask web app, containerized with Docker, and deployed to AWS ECS. The system takes an image, detects the face, predicts emotion, and displays it on a web page.

---

## 📌 Table of Contents

- 🧠 Deep Learning Model

- 🧪 Notebook & Training
   
- 🚀 Flask Web Application
   
- 📦 Dockerization
  
- ☁️ AWS ECS Deployment
   
- 🚀 Usage  


---

## 🧠 Deep Learning Model

We use a **Convolutional Neural Network (CNN)** to classify facial expressions.  

The emotions recognized include: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


The model is trained on grayscale face crops resized to **48×48 pixels**. The final trained model is saved as: emotion_model.pth


---

## 🧪 Notebook & Training

The Jupyter Notebook contains the exploratory code and training logic.

To train the model yourself:

1. Load the dataset of facial images labeled with emotion.
   
2. Build the CNN model (`FER_CNN` class).
   
3. Train the model on GPU/CPU.
   
4. Save the trained model.

📝 After training, the trained model is saved as `emotion_model.pth` for inference in the Flask app.

---

## 🚀 Flask Web Application

The web server (`app.py`) uses:

- **Flask** to serve web pages
  
- **OpenCV** to detect faces in uploaded images
  
- **PyTorch** to perform emotion prediction


### Key Features

- Upload an image through the web interface
  
- Detect faces using Haar Cascade
  
- Predict emotion based on the trained model
  
- Show the uploaded image and predicted emotion


### Start Flask Locally


pip install -r requirements.txt

python app.py

Then open http://localhost:5000

---

## 📦 Dockerization

This project uses a Dockerfile to create a container for the app.

<img width="913" height="366" alt="image" src="https://github.com/user-attachments/assets/4560027f-2ec6-4ab8-aedb-519d6f60e5cf" />

---

## AWS Deployment

<img width="878" height="576" alt="image" src="https://github.com/user-attachments/assets/00e35363-e5b4-4e93-bb10-001b39ab3ff3" />

<img width="701" height="298" alt="image" src="https://github.com/user-attachments/assets/aab39967-5fb5-4f8e-a8b3-29c998fb3466" />

---

## 🚀 Usage

Clone the repo

Configure AWS CLI

Build & test locally

Push Docker to AWS

Deploy to ECS








