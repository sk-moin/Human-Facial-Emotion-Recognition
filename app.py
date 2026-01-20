import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# -------------------- Flask App --------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Device (CPU ONLY) --------------------
device = torch.device("cpu")

# -------------------- Labels --------------------
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(emotion_labels)

# -------------------- Model --------------------
class FER_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------- Load Model --------------------
model = FER_CNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location="cpu"))
model.eval()

# -------------------- Face Detector --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face = gray[y:y+h, x:x+w]
            else:
                face = gray

            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 1, 48, 48))
            face = torch.tensor(face, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(face)
                prediction = emotion_labels[outputs.argmax(1).item()]

    return render_template("index.html", prediction=prediction, image_path=image_path)

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
