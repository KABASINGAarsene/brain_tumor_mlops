import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Setting up our paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_master_model.h5")

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