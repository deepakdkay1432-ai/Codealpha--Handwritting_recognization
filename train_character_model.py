import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "character_model.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 26

# -----------------------------
# Load EMNIST Letters
# -----------------------------
print("Loading EMNIST Letters dataset...")

(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    # EMNIST images can have orientation differences.
    # Rotate and flip to make them visually upright.
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)

    # EMNIST labels are 1-26.
    label = label - 1

    return image, label


ds_train = (
    ds_train
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

ds_test = (
    ds_test
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# -----------------------------
# CNN Model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
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
print("\nTraining character recognition model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# Evaluation
# -----------------------------
test_loss, test_accuracy = model.evaluate(
    ds_test,
    verbose=0
)

print(
    f"\nCharacter Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# -----------------------------
# Save Model
# -----------------------------
model.save(MODEL_PATH)

print("\nCharacter model saved successfully:")
print(MODEL_PATH)
