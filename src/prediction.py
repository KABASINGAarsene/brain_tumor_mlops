import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Loading the Model 
# We use a relative path so it finds the model no matter whose computer it runs on
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
    # Preparing and formatting the Image
    # Resize the incoming image to exactly match the 128x128 training size
    img = image.load_img(img_path, target_size=(128, 128))
    
    # Converting the image to a mathematical array
    img_array = image.img_to_array(img)
    
    # The model expects a "batch" of images, so we put our 1 image inside an empty list-like dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Asking for the Prediction
    prediction = model.predict(img_array)
    
    #  Translating the Math into Medicine
    # Our model uses a Sigmoid activation, meaning it outputs a single probability between 0 and 1
    probability = prediction[0][0]
    
    if probability > 0.5:
        diagnosis = "Tumor Detected"
        # If it's 0.85, that means 85% confident it's a tumor
        confidence = float(probability * 100) 
    else:
        diagnosis = "No Tumor (Healthy)"
        # If it's 0.15, that means it is 85% confident it's healthy (100 - 15)
        confidence = float((1 - probability) * 100)
        
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2)
    }

# Quick test to make sure the script runs when executed directly
if __name__ == "__main__":
    print("Prediction script is ready to serve!")