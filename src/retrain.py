import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import os

# Setting up our paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_master_model.h5")
DB_CSV_PATH = os.path.join(BASE_DIR, "database", "retrain_log.csv")

#  MONKEY PATCHING
_original_random_flip_init = tf.keras.layers.RandomFlip.__init__

def _patched_random_flip_init(self, *args, **kwargs):
    kwargs.pop("data_format", None)
    _original_random_flip_init(self, *args, **kwargs)

tf.keras.layers.RandomFlip.__init__ = _patched_random_flip_init
# --------------------------------------------

def preprocess_new_data(image_paths, numeric_label):
    x_new = []
    y_new = []
    
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        x_new.append(img_array)
        y_new.append(numeric_label)
        
    return np.array(x_new), np.array(y_new)

def retrain_model(image_paths, label_name):
    print("Loading existing model for retraining...")
    
    # Load normally - the patch will silently protect it!
    model = tf.keras.models.load_model(MODEL_PATH)
    
    numeric_label = 1 if "Tumor" in label_name else 0
    x_new, y_new = preprocess_new_data(image_paths, numeric_label)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Retraining model on {len(image_paths)} new images...")
    model.fit(x_new, y_new, epochs=2, batch_size=2)
    model.save(MODEL_PATH)
    
    return f"Successfully retrained the model on {len(image_paths)} images!"

def log_to_database(file_paths, label):
    os.makedirs(os.path.dirname(DB_CSV_PATH), exist_ok=True)
    
    new_data = pd.DataFrame({
        "file_path": file_paths,
        "label": [label] * len(file_paths),
        "status": ["retrained"] * len(file_paths)
    })
    
    if os.path.exists(DB_CSV_PATH):
        new_data.to_csv(DB_CSV_PATH, mode='a', header=False, index=False)
    else:
        new_data.to_csv(DB_CSV_PATH, index=False)