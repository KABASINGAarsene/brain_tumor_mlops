# Brain Tumor Diagnostic Assistant - MLOps Pipeline

## Project Overview
This project is an end-to-end Machine Learning web application designed to assist medical professionals in diagnosing brain tumors from MRI scans. It demonstrates a complete, production-ready MLOps pipeline, covering data exploration, model training, continuous training (retraining), load testing, API deployment, and cloud containerization via Docker.

## Project Structure
* **`src/`**: Contains the core production code:
  * `app.py`: The FastAPI backend server.
  * `frontend.py`: The Streamlit User Interface.
  * `prediction.py`: Loads the model and processes incoming image arrays.
  * `retrain.py`: Handles data preprocessing, model recompilation, and database logging for the Continuous Training pipeline.
* **`models/`**: Houses the finalized, trained model (`brain_tumor_master_model.h5`).
* **`data/`**: Contains the training dataset and a `test/` folder with sample MRI images for evaluating the live application.
* **`database/`**: Auto-generated local database storage containing uploaded training images and the `retrain_log.csv` tracker.
* **`notebook/`**: Contains the original Jupyter Notebook detailing the Exploratory Data Analysis, Data Augmentation, and Deep Learning model training.
* **`locustfile.py`**: Performance testing script to simulate concurrent user traffic.
* **`Dockerfile` & `start.sh`**: Configuration files for containerization and cloud deployment.
* **`requirements.txt`**: List of all required Python dependencies.

---

## Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/brain-tumor-mlops.git](https://github.com/your-username/brain-tumor-mlops.git)
cd brain-tumor-mlops
```

**2. Install Required Dependencies**
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

---

## Running the Application Locally

The application consists of two parts that must run simultaneously: the FastAPI backend and the Streamlit frontend.

### Option A: Using the Startup Script (Mac/Linux/Git Bash)
```bash
chmod +x start.sh
./start.sh
```

### Option B: Manual Startup (Two Terminals)

**Terminal 1 (Backend API):**
```bash
python src/app.py
```

**Terminal 2 (Frontend UI):**
```bash
streamlit run src/frontend.py
```
*The UI will be accessible at http://localhost:8501.*

---

## Advanced MLOps Features

###  Continuous Training (CT) Pipeline
This application features a fully integrated retraining loop. If the model makes an incorrect prediction, a medical professional can use the "Suggestion Box" in the UI to upload the misclassified image along with the correct diagnosis.

* The image is permanently saved to the `database/images/` directory.
* The action is logged in `database/retrain_log.csv`.
* The AI dynamically loads the master `.h5` model, compiles it with a low learning rate, trains on the newly supplied data, and overwrites the model file—learning from its mistakes in real-time.

###  Performance & Load Testing
To ensure the FastAPI server can handle high-traffic environments, a `locustfile.py` is included to simulate user load.

1. Ensure the FastAPI server is running.
2. In a new terminal, run: `python -m locust -f locustfile.py`
3. Open http://localhost:8089 to access the Locust dashboard and simulate hundreds of concurrent API requests.

###  Cloud Deployment (Docker)
This project is fully containerized for deployment on cloud platforms like Render, AWS, or Google Cloud.

```bash
# Build the Docker image
docker build -t brain-tumor-ai .

# Run the Docker container
docker run -p 8000:8000 -p 8501:8501 brain-tumor-ai
```