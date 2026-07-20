import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Page Setup
st.set_page_config(page_title="Brain Tumor AI Diagnostic Tool", layout="centered")
API_URL = "http://127.0.0.1:8000"

st.title(" Brain Tumor Diagnostic Assistant")
st.markdown("Upload an MRI scan to receive an AI-assisted diagnostic prediction.")

# Showing if the system is currently "up" or "down"
st.sidebar.header("System Status")
try:
    # Pinging the FastAPI server to see if it responds
    response = requests.get(API_URL)
    if response.status_code == 200:
        st.sidebar.success("API Server: ONLINE ")
    else:
        st.sidebar.warning("API Server: UNSTABLE ")
except requests.exceptions.ConnectionError:
    st.sidebar.error("API Server: OFFLINE  (Please start the backend)")

# Showing data visualizations telling a story about the dataset
st.header(" Dataset & Model Overview")
st.markdown("Our Master Model was trained on a specialized dataset. As shown below, the training data was naturally imbalanced (more sick patients than healthy patients). We utilized Data Augmentation to ensure the AI learned to identify rare cases accurately.")

# Recreating the Data Distribution Graph from our EDA
data = {'Class': ['Tumor', 'No Tumor (Healthy)'], 'Count': [155, 98]}
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(df['Class'], df['Count'], color=['#ff9999', '#66b3ff'])
ax.set_ylabel("Number of MRI Scans")
ax.set_title("Original Training Data Distribution")

# Add numbers on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom')

st.pyplot(fig)
st.divider()

# Place to upload one image to get a prediction ---
st.header("🩺 Patient MRI Upload")
uploaded_file = st.file_uploader("Choose an MRI image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Showing the doctor the image they just uploaded
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", width=350)

    # The "Order" Button
    if st.button("Run AI Diagnosis"):
        with st.spinner("Analyzing scan..."):
            # Packaging the image to send to our FastAPI Waiter
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Sending the POST request to the API
                res = requests.post(f"{API_URL}/predict", files=files)
                
                if res.status_code == 200:
                    result = res.json()
                    diagnosis = result.get("diagnosis")
                    confidence = result.get("confidence")
                    
                    # Displayin the results beautifully
                    st.subheader("Results:")
                    if "Tumor Detected" in diagnosis:
                        st.error(f"**Diagnosis:** {diagnosis}")
                    else:
                        st.success(f"**Diagnosis:** {diagnosis}")
                        
                    st.info(f"**AI Confidence:** {confidence}%")
                else:
                    st.error(f"Server Error: {res.text}")
                    
            except Exception as e:
                st.error("Could not reach the AI. Please ensure the FastAPI server is running.")

st.divider()

# Uploading Bulk Data and Triggering Retraining
st.header(" Model Retraining (Suggestion Box)")
st.markdown("Upload a batch of new, verified MRI scans to update the AI model. Data is saved to the system database prior to training.")

retrain_label = st.selectbox("What is the true diagnosis for these new images?", ["Tumor", "No Tumor (Healthy)"])
bulk_files = st.file_uploader("Upload multiple MRIs for retraining...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button(" Trigger Model Retraining"):
    if len(bulk_files) == 0:
        st.warning("Please upload at least one image to retrain.")
    else:
        with st.spinner("Saving to database and retraining AI Model... This may take a minute..."):
            # Packaging the files to send to FastAPI
            files_to_send = [
                ("files", (file.name, file.getvalue(), file.type)) for file in bulk_files
            ]
            data = {"label": retrain_label}
            
            try:
                res = requests.post(f"{API_URL}/retrain", data=data, files=files_to_send)
                
                if res.status_code == 200:
                    result = res.json()
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f" {result['message']}")
                        st.info("The Master Model has been updated with the new knowledge!")
                else:
                    st.error(f"Server Error: {res.text}")
            except Exception as e:
                st.error("Could not reach the AI server.")