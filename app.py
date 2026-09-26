from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np

from ultralytics import YOLO
from tensorflow.keras.models import load_model


app = Flask(__name__)


# =====================================================
# LOAD MODELS
# =====================================================

print("Loading YOLO face model...")

face_model = YOLO(
    "models/yolov11n-face.pt"
)


print("Loading emotion model...")

emotion_model = load_model(
    "models/emotion_model.h5"
)


# =====================================================
# EMOTION LABELS
# =====================================================

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# =====================================================
# DETECT EMOTION
# =====================================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    if "image" not in request.files:

        return jsonify({
            "error": "No image received."
        }), 400


    file = request.files["image"]


    image_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )


    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        return jsonify({
            "error": "Invalid image."
        }), 400


    # =================================================
    # YOLO FACE DETECTION
    # =================================================

    results = face_model.predict(
        frame,
        imgsz=320,
        conf=0.40,
        verbose=False
    )


    detections = []


    for result in results:

        if result.boxes is None:
            continue


        boxes = result.boxes.xyxy.cpu().numpy()


        for box in boxes:

            x1, y1, x2, y2 = box.astype(int)


            # Keep coordinates inside image

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            face = frame[
                y1:y2,
                x1:x2
            ]


            if face.size == 0:
                continue


            # =========================================
            # PREPARE FACE FOR EMOTION MODEL
            # =========================================

            gray = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )


            gray = cv2.resize(
                gray,
                (48, 48)
            )


            gray = gray.astype(
                "float32"
            ) / 255.0


            gray = np.expand_dims(
                gray,
                axis=-1
            )


            gray = np.expand_dims(
                gray,
                axis=0
            )


            # =========================================
            # EMOTION PREDICTION
            # =========================================

            prediction = emotion_model.predict(
                gray,
                verbose=0
            )


            emotion_index = int(
                np.argmax(prediction[0])
            )


            emotion = EMOTIONS[
                emotion_index
            ]


            confidence = float(
                prediction[0][emotion_index]
            ) * 100


            detections.append({

                "x": int(x1),

                "y": int(y1),

                "width": int(x2 - x1),

                "height": int(y2 - y1),

                "emotion": emotion,

                "confidence": round(
                    confidence,
                    1
                )

            })


    return jsonify({

        "faces": detections

    })


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )