#  Brain Tumor Diagnostic Assistant - MLOps Pipeline

## Project Overview
This project is an end-to-end Machine Learning web application designed to assist medical professionals in diagnosing brain tumors from MRI scans. It demonstrates a complete MLOps pipeline, from data exploration and model training to API deployment and a user-facing frontend.

## Project Structure
*   **`notebook/`**: Contains the original Jupyter Notebook (`brain_tumor_eda.ipynb`) detailing the Exploratory Data Analysis, Data Augmentation, and Deep Learning model training (MobileNetV2).
*   **`models/`**: Houses the finalized, trained model (`brain_tumor_master_model.h5`).
*   **`data/`**: Contains the training dataset and a `test/` folder with sample MRI images for evaluating the live application.
*   **`src/`**: Contains the production code:
    *   `prediction.py`: Loads the model and processes image arrays.
    *   `app.py`: The FastAPI backend server.
    *   `frontend.py`: The Streamlit User Interface.

## Installation & Setup

**1. Install Required Dependencies**
Ensure you have Python installed, then install the necessary libraries:
```bash
pip install tensorflow numpy pillow fastapi uvicorn python-multipart streamlit requests pandas matplotlib