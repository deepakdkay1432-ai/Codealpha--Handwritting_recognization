import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load MNIST Dataset
# -----------------------------
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values: 0-255 -> 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

# -----------------------------
# CNN Model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Training
# -----------------------------
print("\nTraining digit recognition model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Evaluation
# -----------------------------
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
model.save(MODEL_PATH)

print(f"\nModel saved successfully:")
print(MODEL_PATH)
