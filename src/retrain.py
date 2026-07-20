import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import os

# Setting up our paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_master_model.h5")
DB_CSV_PATH = os.path.join(BASE_DIR, "database", "retrain_log.csv")

def preprocess_new_data(image_paths, numeric_label):
    """
    Data Preprocessing of the uploaded data.
    Resizes images to 128x128 and converts them to mathematical arrays.
    """
    x_new = []
    y_new = []
    
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        x_new.append(img_array)
        y_new.append(numeric_label)
        
    return np.array(x_new), np.array(y_new)

def retrain_model(image_paths, label_name):
    """
     Retraining - uses custom model as a pre-trained model.
    """
    print("Loading existing model for retraining...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Converting text label to number (1 for Tumor, 0 for Healthy)
    numeric_label = 1 if "Tumor" in label_name else 0
    
    # Preprocessing the data
    x_new, y_new = preprocess_new_data(image_paths, numeric_label)
    
    # Compiling with a very low learning rate to protect existing knowledge
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Retraining the model for 2 epochs
    print(f"Retraining model on {len(image_paths)} new images...")
    model.fit(x_new, y_new, epochs=2, batch_size=2)
    
    # Saving the updated model
    model.save(MODEL_PATH)
    
    return f"Successfully retrained the model on {len(image_paths)} images!"

def log_to_database(file_paths, label):
    """
    Saving to Database.
    Logs the retraining event to a CSV file.
    """
    os.makedirs(os.path.dirname(DB_CSV_PATH), exist_ok=True)
    
    new_data = pd.DataFrame({
        "file_path": file_paths,
        "label": [label] * len(file_paths),
        "status": ["retrained"] * len(file_paths)
    })
    
    # Appending to existing CSV or creating a new one
    if os.path.exists(DB_CSV_PATH):
        new_data.to_csv(DB_CSV_PATH, mode='a', header=False, index=False)
    else:
        new_data.to_csv(DB_CSV_PATH, index=False)