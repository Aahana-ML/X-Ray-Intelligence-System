# 🩻 X-Ray Intelligence System

An AI-assisted chest X-ray analysis system for multi-label abnormality classification using deep learning, with Grad-CAM explainability, FastAPI, Docker, and a web interface.

## 🚀 Live Demo

🔗 **[Open X-Ray Intelligence System](https://x-ray-intelligence-system-1.onrender.com/)**

> ⚠️ This system is for educational and research purposes only and is not a medical diagnostic system.


## 📌 Project Overview

Chest X-rays can contain multiple abnormalities in the same image. Therefore, this project approaches chest X-ray analysis as a **multi-label classification problem**, where a single X-ray can receive multiple abnormality predictions.

The system uses a deep learning model based on **EfficientNetB0** to analyze chest X-ray images and produce independent scores for 14 abnormalities.

It also includes **Grad-CAM explainability**, allowing users to visualize the regions of the X-ray that contributed to the model's prediction.

## 🎯 Problem Statement

The goal of this project is to build an end-to-end AI-assisted system that can:

- Analyze a chest X-ray image.
- Predict multiple possible abnormalities.
- Apply class-specific decision thresholds.
- Provide evaluation metrics for model performance.
- Generate Grad-CAM visualizations for model explainability.
- Expose the trained model through a FastAPI backend.
- Provide a web interface for interacting with the system.
- Package and deploy the complete application using Docker.


## 🩻 Supported Abnormalities

The model performs multi-label classification across 14 chest X-ray abnormalities:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural Thickening
- Pneumonia
- Pneumothorax

Because this is a multi-label classification system, multiple abnormalities can be predicted for a single X-ray.


## 🧠 Model & ML Pipeline

The system uses **EfficientNetB0** as the backbone for multi-label chest X-ray classification.

### Model Architecture

```text
Input X-Ray
     │
     ▼
EfficientNetB0
(ImageNet pretrained)
     │
     ▼
Global Average Pooling
     │
     ▼
Dropout
     │
     ▼
Dense Layer
     │
     ▼
14 Sigmoid Outputs
```

## Training Strategy

The model was developed using the following approach:

- Transfer Learning
EfficientNetB0 was initialized with ImageNet pretrained weights.
The pretrained backbone was initially frozen while training the classification head.

- Fine-Tuning
The final layers of EfficientNetB0 were unfrozen.

- Batch Normalization layers were kept frozen.
A lower learning rate was used during fine-tuning.

- Data Augmentation
Random rotation
Random translation
Random zoom
Random contrast adjustment

- Multi-Label Classification
Each abnormality has an independent sigmoid output.
Binary Cross Entropy was used as the loss function.

- Threshold Tuning
Instead of using a fixed 0.5 threshold for every abnormality, class-specific thresholds were selected using the validation set.
These thresholds were then applied to the held-out test set.


## Input Processing

Chest X-ray images are:

- Decoded as RGB images
- Resized to 224 × 224
- Converted to floating-point tensors
- Passed through the EfficientNetB0 pipeline

The final model produces 14 independent outputs, one for each supported abnormality.


```
## 📊 Evaluation & Results

The model was evaluated on a held-out test set using metrics suited for multi-label classification and class imbalance.

### Evaluation Metrics

- **ROC-AUC** — measures how well the model ranks positive cases above negative cases.
- **PR-AUC / Average Precision** — provides additional insight when positive cases are relatively rare.
- **Precision** — proportion of predicted positives that are actually positive.
- **Recall** — proportion of actual positives identified by the model.
- **F1-score** — harmonic mean of precision and recall.

### Final Test Performance

The final model achieved:

| Metric | Score |
|---|---:|
| Macro ROC-AUC | **0.7314** |

The ROC-AUC was also evaluated separately for each abnormality to understand how model performance varied across classes.

Because the dataset contains substantial class imbalance, **ROC-AUC was not considered in isolation**. Precision-Recall curves and Average Precision were also examined for individual abnormalities.

### Threshold Optimization

The default classification threshold of `0.5` was not suitable for all abnormalities.

Therefore, thresholds were optimized independently using the validation set and then applied to the untouched test set.

This allowed the inference system to use a different decision threshold for each abnormality.
```

```
## 🔥 Explainability with Grad-CAM

The system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide a visual explanation of the model's predictions.

For a selected abnormality, Grad-CAM generates a heatmap showing the regions of the X-ray that contributed to the model's score for that class.

### How it works

```text
Input X-Ray
     │
     ▼
EfficientNetB0
     │
     ▼
Target Convolutional Layer
     │
     ▼
Gradients for Selected Class
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Heatmap Overlay
```


## ⚡ FastAPI Backend

The trained model is served through a **FastAPI REST API**.

The backend handles image preprocessing, model inference, threshold-based predictions, and Grad-CAM generation.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web application |
| `POST` | `/predict` | Analyzes an uploaded X-ray and returns abnormality scores |
| `POST` | `/explain` | Generates a Grad-CAM visualization for a selected abnormality |

### Prediction Response

The `/predict` endpoint returns an independent score and detection result for each abnormality.

Example:

```json
{
    "Effusion": {
        "score": 0.3958,
        "detected": true
    },
    "Pneumonia": {
        "score": 0.0267,
        "detected": false
    }
}
```

## 🌐 Web Interface

The project includes a lightweight web interface built with:

- HTML
- CSS
- JavaScript

The interface provides a simple workflow for interacting with the deployed model.

### User Workflow

```text
Upload Chest X-Ray
        │
        ▼
   Analyze X-Ray
        │
        ▼
 View Predictions
        │
        ▼
 Select Abnormality
        │
        ▼
 Generate Grad-CAM
        │
        ▼
 View Attention Heatmap
```

## 🐳 Docker & Cloud Deployment

The complete application is containerized using **Docker**, allowing the backend, trained model, configuration files, and frontend to run as a single deployable application.

### Docker Architecture

```text
Docker Container
│
├── FastAPI Backend
├── TensorFlow Model
├── Grad-CAM
├── Configuration
└── Web Frontend
```

## 🐳 Docker & Cloud Deployment

The complete application is containerized using **Docker**, allowing the backend, trained model, configuration files, and frontend to run as a single deployable application.



### Part 10 — Project Structure

## 📁 Project Structure

```text
X-Ray-Intelligence-System/
│
├── app/
│   ├── main.py
│   ├── inference.py
│   └── gradcam.py
│
├── config/
│   ├── class_names.json
│   └── xray_thresholds.json
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   └── xray_final.keras
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore

```
## Main Components
- app/main.py — FastAPI application and API endpoints
- app/inference.py — model loading, preprocessing, and prediction
- app/gradcam.py — Grad-CAM generation
- frontend/ — web interface
- config/ — class names and optimized thresholds
- models/ — trained EfficientNetB0 model
- Dockerfile — container configuration


## 📚 Dataset

This project uses the **NIH ChestX-ray14** dataset for training and evaluation.

The dataset contains chest X-ray images with multiple possible findings, making it suitable for the multi-label classification approach used in this project.

### Dataset Source

The original dataset is provided by the National Institutes of Health:

🔗 [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

A Kaggle version of the dataset was used for model development and GPU-based training.

The dataset is **not included in this repository** due to its size and licensing considerations.


## 🛠️ Tech Stack

### Machine Learning
- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Scikit-learn

### Backend
- FastAPI
- Uvicorn

### Explainability
- Grad-CAM
- Matplotlib

### Frontend
- HTML
- CSS
- JavaScript

### Deployment
- Docker
- Render

### Version Control
- Git
- GitHub
  

## ⚠️ Limitations & Disclaimer

This project is intended for **educational and research purposes only**.

It is not a clinically validated medical diagnostic system and should not be used for diagnosis or clinical decision-making.

### Limitations

- The dataset contains substantial class imbalance.
- Some abnormalities have relatively few training examples.
- Model performance may vary on images from different datasets or populations.
- The reported evaluation results are specific to the held-out test set used in this project.
- The class-specific thresholds were optimized using the validation set.
- Grad-CAM visualizations represent model attention and should not be interpreted as clinically validated localization.
- The model's output scores should not be interpreted as calibrated clinical probabilities.


## 👩‍💻 Author

**Aahana Panwar**

Machine Learning & Deep Learning Project

🔗 [GitHub](https://github.com/Aahana-ML)
