import os
import shutil
from fastapi import FastAPI, UploadFile, File
import uvicorn
from src.prediction import predict_mri # Importing your prediction script
from typing import List
from fastapi import Form
from src.retrain import retrain_model, log_to_database 

# Initializing the API application
app = FastAPI(
    title="Brain Tumor Diagnosis API", 
    description="API for predicting brain tumors from MRI scans."
)

@app.get("/")
def home():
    """
    A simple health check to ensure the server is running.
    """
    return {"message": "Welcome to the Brain Tumor API. Send a POST request with an image to /predict to get a diagnosis."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, saves it temporarily, runs it through the ML model, 
    and returns the diagnosis and confidence score.
    """
    # Creating a temporary folder to store the incoming image
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    
    # Saving the uploaded file to the computer so the model can read it
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Handing the saved image path to the prediction function
    try:
        # Running the prediction
        result = predict_mri(file_path)
    except Exception as e:
        return {"error": f"Failed to process image. Details: {str(e)}"}
    finally:
        # Deleting the temporary image so the hard drive doesn't get full
        if os.path.exists(file_path):
            os.remove(file_path)
            
    # Returning the JSON result back to the user/web app
    return result

@app.post("/retrain")
async def retrain_endpoint(label: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Accepts bulk images, saves them to the database, and triggers retraining.
    """
    # Creating the permanent database folder for images
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "images")
    os.makedirs(db_dir, exist_ok=True)
    
    saved_file_paths = []
    
    # Saving all uploaded files to the database
    for file in files:
        file_path = os.path.join(db_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_file_paths.append(file_path)
        
    try:
        # Logging the files to the CSV Database
        log_to_database(saved_file_paths, label)
        
        # Triggering the retraining script
        message = retrain_model(saved_file_paths, label)
        return {"message": message, "status": "Success"}
    except Exception as e:
        return {"error": f"Retraining failed: {str(e)}"}
    

if __name__ == "__main__":
    print("Starting the API server...")
    # This runs the server locally on the machine at port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)