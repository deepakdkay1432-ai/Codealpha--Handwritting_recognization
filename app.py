import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Recognition",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    digit_model = tf.keras.models.load_model(
        "models/digit_model.keras"
    )

    character_model = tf.keras.models.load_model(
        "models/character_model.keras"
    )

    return digit_model, character_model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Convert drawing into the 28x28 format
    expected by the CNN models.
    """

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST/EMNIST size
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy
    image_array = np.array(image).astype("float32")

    # Normalize
    image_array = image_array / 255.0

    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# DIGIT PREDICTION
# ============================================================

def predict_digit(image, model):

    processed_image = preprocess_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )

    return predicted_class, confidence


# ============================================================
# CHARACTER PREDICTION
# ============================================================

def predict_character(image, model):

    processed_image = preprocess_image(image)

    # EMNIST preprocessing
    # The training script rotates/flips EMNIST images.
    image_tensor = tf.convert_to_tensor(
        processed_image
    )

    image_tensor = tf.transpose(
        image_tensor,
        perm=[0, 2, 1, 3]
    )

    image_tensor = tf.image.flip_left_right(
        image_tensor
    )

    prediction = model.predict(
        image_tensor,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )

    # 0 -> A, 1 -> B, ..., 25 -> Z
    predicted_character = chr(
        predicted_class + ord("A")
    )

    return predicted_character, confidence


# ============================================================
# LOAD MODELS
# ============================================================

try:

    digit_model, character_model = load_models()

except Exception as e:

    st.error(
        "Models could not be loaded."
    )

    st.info(
        "Make sure these files exist:\n\n"
        "`models/digit_model.keras`\n\n"
        "`models/character_model.keras`"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("✍️ Handwritten Recognition")

st.write(
    "Draw a handwritten digit or character "
    "and let the CNN model recognize it."
)


# ============================================================
# SELECT MODE
# ============================================================

mode = st.radio(
    "Select Recognition Type",
    [
        "Digit (0-9)",
        "Character (A-Z)"
    ],
    horizontal=True
)


# ============================================================
# DRAWING CANVAS
# ============================================================

st.subheader("Draw Here")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas"
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Predict",
    use_container_width=True
):

    if canvas_result.image_data is None:

        st.warning(
            "Please draw something first."
        )

    else:

        # Convert canvas to PIL image
        canvas_image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Check whether canvas contains drawing
        image_array = np.array(
            canvas_image.convert("L")
        )

        if np.max(image_array) < 10:

            st.warning(
                "Please draw a digit or character first."
            )

        else:

            # ------------------------------------------------
            # DIGIT MODE
            # ------------------------------------------------

            if mode == "Digit (0-9)":

                prediction, confidence = predict_digit(
                    canvas_image,
                    digit_model
                )

                st.success(
                    f"Prediction: **{prediction}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


            # ------------------------------------------------
            # CHARACTER MODE
            # ------------------------------------------------

            else:

                prediction, confidence = predict_character(
                    canvas_image,
                    character_model
                )

                st.success(
                    f"Prediction: **{prediction}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )


# ============================================================
# INSTRUCTIONS
# ============================================================

with st.expander("ℹ️ How to use"):

    st.write(
        """
        1. Select **Digit (0-9)** or **Character (A-Z)**.
        2. Draw clearly inside the canvas.
        3. Click **Predict**.
        4. The CNN model will display the predicted class
           and confidence score.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CodeAlpha Task 3 — Handwritten Recognition | "
    "Built with TensorFlow & Streamlit"
)
