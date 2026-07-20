#  Brain Tumor Diagnostic Assistant - MLOps Pipeline

## Project Overview
This project is an end-to-end Machine Learning web application designed to assist medical professionals in diagnosing brain tumors from MRI scans. It demonstrates a complete, production-ready MLOps pipeline, covering data exploration, model training, continuous training (retraining), load testing, API deployment, and cloud containerization via Docker.

## Project Structure
*   **`src/`**: Contains the core production code:
    *   `app.py`: The FastAPI backend server.
    *   `frontend.py`: The Streamlit User Interface.
    *   `prediction.py`: Loads the model and processes incoming image arrays.
    *   `retrain.py`: Handles data preprocessing, model recompilation, and database logging for the Continuous Training pipeline.
*   **`models/`**: Houses the finalized, trained model (`brain_tumor_master_model.h5`).
*   **`data/`**: Contains the training dataset and a `test/` folder with sample MRI images for evaluating the live application.
*   **`database/`**: Auto-generated local database storage containing uploaded training images and the `retrain_log.csv` tracker.
*   **`notebook/`**: Contains the original Jupyter Notebook detailing the Exploratory Data Analysis, Data Augmentation, and Deep Learning model training.
*   **`locustfile.py`**: Performance testing script to simulate concurrent user traffic.
*   **`Dockerfile` & `start.sh`**: Configuration files for containerization and cloud deployment.
*   **`requirements.txt`**: List of all required Python dependencies.

---

## Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/brain-tumor-mlops.git](https://github.com/your-username/brain-tumor-mlops.git)
cd brain-tumor-mlops