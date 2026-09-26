import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xray_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "config" / "class_names.json"
THRESHOLDS_PATH = BASE_DIR / "config" / "xray_thresholds.json"


# Load model once when the application starts
model = load_model(MODEL_PATH)


# Load class names
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


# Load thresholds
with open(THRESHOLDS_PATH, "r") as file:
    thresholds = json.load(file)


async def preprocess_image(file):

    contents = await file.read()

    image = tf.image.decode_image(
        contents,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        (224, 224)
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


async def predict_image(file):

    image = await preprocess_image(file)

    predictions = model.predict(
        image,
        verbose=0
    )[0]

    results = {}

    for label, score in zip(class_names, predictions):

        threshold = thresholds[label]

        results[label] = {
            "score": float(score),
            "detected": bool(score >= threshold)
        }

    return image, results