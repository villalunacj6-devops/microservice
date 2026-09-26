from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np


app = Flask(__name__)


# =====================================================
# GLOBAL MODELS
# =====================================================

face_model = None
emotion_model = None


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
# LOAD MODELS
# =====================================================

def load_models():

    global face_model
    global emotion_model

    # ---------------------------------------------
    # Load YOLO only once
    # ---------------------------------------------

    if face_model is None:

        print("Loading YOLO face model...", flush=True)

        from ultralytics import YOLO

        face_model = YOLO(
            "models/model.pt"
        )

        print(
            "YOLO face model loaded.",
            flush=True
        )


    # ---------------------------------------------
    # Load emotion model only once
    # ---------------------------------------------

    if emotion_model is None:

        print(
            "Loading emotion model...",
            flush=True
        )

        from tensorflow.keras.models import load_model

        emotion_model = load_model(
            "models/emotion_model.h5",
            compile=False
        )

        print(
            "Emotion model loaded.",
            flush=True
        )


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

    try:

        # ---------------------------------------------
        # Load AI models
        # ---------------------------------------------

        load_models()


        # ---------------------------------------------
        # Check uploaded image
        # ---------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "error": "No image received."
            }), 400


        file = request.files["image"]


        # ---------------------------------------------
        # Convert image to NumPy
        # ---------------------------------------------

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


        # ---------------------------------------------
        # YOLO FACE DETECTION
        # ---------------------------------------------

        results = face_model.predict(
            source=frame,
            imgsz=320,
            conf=0.40,
            device="cpu",
            verbose=False
        )


        detections = []


        # ---------------------------------------------
        # Process detected faces
        # ---------------------------------------------

        for result in results:

            if result.boxes is None:
                continue


            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )


            for box in boxes:

                x1, y1, x2, y2 = (
                    box.astype(int)
                )


                # -------------------------------------
                # Keep coordinates inside image
                # -------------------------------------

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    frame.shape[1],
                    x2
                )

                y2 = min(
                    frame.shape[0],
                    y2
                )


                # -------------------------------------
                # Extract face
                # -------------------------------------

                face = frame[
                    y1:y2,
                    x1:x2
                ]


                if face.size == 0:
                    continue


                # -------------------------------------
                # Convert to grayscale
                # -------------------------------------

                gray = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2GRAY
                )


                # -------------------------------------
                # Resize to model input
                # -------------------------------------

                gray = cv2.resize(
                    gray,
                    (48, 48)
                )


                # -------------------------------------
                # Normalize
                # -------------------------------------

                gray = (
                    gray.astype(
                        "float32"
                    ) / 255.0
                )


                # -------------------------------------
                # Add channels
                # Shape:
                # (48,48)
                # →
                # (48,48,1)
                # →
                # (1,48,48,1)
                # -------------------------------------

                gray = np.expand_dims(
                    gray,
                    axis=-1
                )


                gray = np.expand_dims(
                    gray,
                    axis=0
                )


                # -------------------------------------
                # Emotion prediction
                # -------------------------------------

                prediction = (
                    emotion_model.predict(
                        gray,
                        verbose=0
                    )
                )


                emotion_index = int(
                    np.argmax(
                        prediction[0]
                    )
                )


                emotion = EMOTIONS[
                    emotion_index
                ]


                confidence = float(
                    prediction[0][
                        emotion_index
                    ]
                ) * 100


                # -------------------------------------
                # Save detection
                # -------------------------------------

                detections.append({

                    "x": int(x1),

                    "y": int(y1),

                    "width": int(
                        x2 - x1
                    ),

                    "height": int(
                        y2 - y1
                    ),

                    "emotion": emotion,

                    "confidence": round(
                        confidence,
                        1
                    )

                })


        # ---------------------------------------------
        # Return results
        # ---------------------------------------------

        return jsonify({

            "faces": detections

        })


    except Exception as e:

        print(
            "DETECTION ERROR:",
            str(e),
            flush=True
        )

        return jsonify({

            "error": str(e)

        }), 500


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )