# Uses an official, lightweight Python image
FROM python:3.9-slim

# Sets the working directory inside the container
WORKDIR /app

# Copies the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies all your code, models, and data into the container
COPY . .

# Makes the startup script executable
RUN chmod +x start.sh

# Opens the ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Command to run when the container starts
CMD ["./start.sh"]