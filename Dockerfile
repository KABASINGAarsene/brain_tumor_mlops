# Uses  preferred Python 3.9 slim image
FROM python:3.9-slim

# Sets the working directory
WORKDIR /app


RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copies the requirements file and installs dependencies

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies all the code, models, and data into the container
COPY --chown=user . .

# Starts ONLY the FastAPI backend on Hugging Face's on port (7860)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]