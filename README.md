# ✍️ Handwritten Recognition

A deep learning-based handwritten recognition application developed as part of the **CodeAlpha Internship — Task 3**.

The application uses **Convolutional Neural Networks (CNNs)** to recognize handwritten **digits (0–9)** and **English uppercase characters (A–Z)** through an interactive Streamlit interface.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
_Add your deployed Streamlit URL here_

---

## 📌 Project Overview

Handwritten recognition is an image classification problem where a deep learning model learns to identify handwritten characters from images.

This project implements two separate CNN models:

- 🔢 **Digit Recognition Model** — recognizes handwritten digits from 0–9.
- 🔤 **Character Recognition Model** — recognizes handwritten English uppercase characters from A–Z.

Users can draw a digit or character directly on the application canvas, and the trained model predicts the result along with its confidence score.

---

## ✨ Features

- 🔢 Handwritten digit recognition (0–9)
- 🔤 Handwritten character recognition (A–Z)
- 🧠 CNN-based image classification
- 🎨 Interactive drawing canvas
- 📊 Prediction confidence score
- ⚡ Fast predictions
- 🌐 Streamlit web application
- 📱 Simple and user-friendly interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- Streamlit
- Streamlit Drawable Canvas
- Convolutional Neural Networks (CNN)

---

## 📚 Datasets

### MNIST

The **MNIST** dataset is used for handwritten digit recognition.

It contains:

- **60,000** training images
- **10,000** test images
- **10** classes
- Grayscale handwritten digit images

Classes:

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### EMNIST Letters

The **EMNIST Letters** dataset is used for handwritten English character recognition.

It contains:

- **88,800** training images
- **14,800** test images
- **26** character classes

The classes represent:

    A, B, C, ..., Z

---

## 🧠 Model Architecture

Both recognition models use a **Convolutional Neural Network (CNN)** architecture.

    Input Image
         ↓
    Conv2D - 32 Filters
         ↓
    MaxPooling
         ↓
    Conv2D - 64 Filters
         ↓
    MaxPooling
         ↓
    Conv2D - 128 Filters
         ↓
    Flatten
         ↓
    Dense - 128 Neurons
         ↓
    Dropout
         ↓
    Softmax Output

### 🔢 Digit Model

The digit model has **10 output classes**:

    0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### 🔤 Character Model

The character model has **26 output classes**:

    A, B, C, ..., Z

---

## 📊 Model Performance

The EMNIST character recognition model achieved approximately:

**92.60% test accuracy**

on the EMNIST Letters test dataset.

The digit recognition model is evaluated on the **MNIST test dataset** during training.

> Note: Actual performance may vary depending on training configuration, preprocessing, and model version.

---

## 📁 Project Structure

    Codealpha--Handwriting_recognization/
    │
    ├── app.py
    ├── train_digit_model.py
    ├── train_character_model.py
    ├── requirements.txt
    ├── README.md
    │
    └── models/
        ├── digit_model.keras
        └── character_model.keras

---

## ⚙️ Installation

### 1. Clone the Repository

    git clone https://github.com/deepakdkay1432-ai/Codealpha--Handwriting_recognization.git

### 2. Move Into the Project Directory

    cd Codealpha--Handwriting_recognization

### 3. Install Dependencies

    pip install -r requirements.txt

---

## ▶️ Run the Application

Start the Streamlit application using:

    streamlit run app.py

The application will open in your web browser.

---

## 🖥️ How to Use

1. Open the Streamlit application.
2. Select **Digit (0–9)** or **Character (A–Z)**.
3. Draw a digit or character on the canvas.
4. Click **Predict**.
5. The application displays the predicted result.
6. View the model's prediction confidence.

---

## 💾 Trained Models

The trained models are stored inside the `models` directory:

- `models/digit_model.keras`
- `models/character_model.keras`

The Streamlit application automatically loads these trained models when it starts.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Implement CNN-based image classification.
- Understand handwritten image preprocessing.
- Train deep learning models using MNIST and EMNIST.
- Build an interactive Streamlit application.
- Integrate trained machine learning models into a web application.
- Deploy a deep learning application for real-world use.

---

## 🔮 Future Improvements

Possible improvements include:

- 📈 Improved model accuracy
- 🖼️ Better image preprocessing
- 🎯 Automatic cropping and centering
- 🔡 Support for lowercase characters
- 📤 Image upload functionality
- 📱 Improved mobile interface
- 🧠 More advanced CNN architectures
- ⚡ Faster inference and optimized model loading

---

## 👨‍💻 Author

**Deepak Kumar Gupta**

**Data Analyst | Python Developer | AI/ML Enthusiast**

---

## 📜 Internship

This project was developed as part of the **CodeAlpha Internship — Task 3**.

---

⭐ If you found this project useful, consider giving the repository a **star**!
