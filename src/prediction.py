import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Setting up our paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_master_model.h5")

# --- MONKEY PATCHES ---
# Fixing RandomFlip data_format error
_original_random_flip_init = tf.keras.layers.RandomFlip.__init__
def _patched_random_flip_init(self, *args, **kwargs):
    kwargs.pop("data_format", None)
    _original_random_flip_init(self, *args, **kwargs)
tf.keras.layers.RandomFlip.__init__ = _patched_random_flip_init

# 2. Fixing GlorotUniform input_axes error
_original_glorot_init = tf.keras.initializers.GlorotUniform.__init__
def _patched_glorot_init(self, *args, **kwargs):
    kwargs.pop("input_axes", None)   # Destroyes the bad input_axes argument
    kwargs.pop("output_axes", None)  # Destroyes output_axes just to be safe
    _original_glorot_init(self, *args, **kwargs)
tf.keras.initializers.GlorotUniform.__init__ = _patched_glorot_init
# ------------------------------

print("Loading the Brain Tumor Master Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict_mri(img_path):
    """
    Takes a patient's MRI image, prepares it for the model, 
    and returns a clean, readable diagnosis.
    """
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    if probability > 0.5:
        diagnosis = "Tumor Detected"
        confidence = float(probability * 100) 
    else:
        diagnosis = "No Tumor (Healthy)"
        confidence = float((1 - probability) * 100)
        
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    print("Prediction script is ready to serve!")