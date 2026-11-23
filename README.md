# Skin and Hair Disease Detection using Deep Learning

A web application that classifies skin and hair conditions from an uploaded photo using three deep learning models built with TensorFlow and served through Flask.

> **Disclaimer:** This is an academic project for demonstration only. It is not a medical device and must not be used to diagnose or treat any condition. Always consult a qualified doctor.

## Overview

The user chooses one of three detectors, enters basic details, and uploads an image. The matching model predicts the condition and the app shows a short description of it.

| Detector | Image type | Classes | Images (train / val / test) | Test accuracy |
|---|---|---|---|---|
| Hair disease | Scalp and hair photos | 5: Alopecia Areata, Folliculitis, Psoriasis, Seborrheic Dermatitis, Tinea Capitis | 4,800 / 600 / 600 | ~99% |
| Skin disease, model 1 | Clinical images | 2: Atopic Dermatitis, Eczema | 2,346 / 292 / 296 | ~83% |
| Skin disease, model 2 | Dermatoscopic images | 3: Basal Cell Carcinoma, Melanocytic Nevi, Melanoma | 3,600 / 450 / 450 | ~89% |

Full training and evaluation details (accuracy and loss curves, confusion matrices, classification reports) are in the notebooks.

## How it works

1. Images are resized to 224 x 224 pixels.
2. Data augmentation (random rotation, horizontal flip, and random width and height changes) increases the variety of the training data.
3. Each dataset is split 80:10:10 into training, validation and test sets.
4. All three models use transfer learning: a frozen EfficientNetB7 base with a small classifier on top (global average pooling, a 32-unit dense layer and a softmax output). They were trained for 20 epochs with early stopping and learning-rate reduction enabled.
5. The Flask app loads the trained models, accepts an uploaded image, returns the predicted class and shows a description of the condition.

## Tech stack

- Python
- TensorFlow / Keras
- Flask
- NumPy, Pillow, scikit-learn, Matplotlib, split-folders
- SQLite (stores the details entered in the web form)
- HTML templates


## Setup and run

1. Clone the repository:
   ```bash
   git clone https://github.com/Bismiyabb/Skin-and-Hair-Diseases-Detection-using-Deep-Learning.git
   cd Skin-and-Hair-Diseases-Detection-using-Deep-Learning
   ```
2. Install the packages imported at the top of `app.py` (Flask, TensorFlow, Pillow, NumPy, OpenCV, imutils, scikit-learn, joblib):
   ```bash
   pip install flask tensorflow pillow numpy opencv-python imutils scikit-learn joblib
   ```
3. Make sure the `models/` folder contains `model1.json`, `model2.json`, `model3.json`, `skin_model.hdf5`, `skin_model2.hdf5` and `model.hdf5`.
4. Create the upload folder if it does not exist: `static/uploads/`.
5. Run the app:
   ```bash
   python app.py
   ```
6. Open `http://127.0.0.1:5000` in your browser.

## Results

- **Hair disease model:** about 99% test accuracy over 5 classes, with training and validation curves close together.
- **Skin model 2 (dermatoscopic):** about 89% test accuracy over 3 classes.
- **Skin model 1 (clinical):** about 83% test accuracy over 2 classes. Training accuracy rose to about 0.89 while validation accuracy stayed near 0.80, which shows some overfitting.

## Known limitations

- The models can only choose between the classes they were trained on. If an unrelated image (for example, a car) is uploaded, the model still returns the closest class instead of rejecting it.
- Skin model 1 overfits its training data.
- Accuracy depends on image quality, lighting and the datasets used. The models are not validated for clinical use.
- The demo stores the form details (name, email, phone, age, gender) and the result in a local SQLite database. It has no authentication and is intended for local use only.

## Future work

- Add an image-validation step that rejects images that are not skin or hair.
- Reduce overfitting in skin model 1 with more data, stronger augmentation, dropout or fine-tuning part of the EfficientNetB7 base.
- Evaluate on more diverse images and skin tones.

