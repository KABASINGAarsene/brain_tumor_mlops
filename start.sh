#!/bin/bash
# Starts the FastAPI backend in the background
uvicorn src.app:app --host 0.0.0.0 --port 8000 &

# Starts the Streamlit frontend
streamlit run src/frontend.py --server.port 8501 --server.address 0.0.0.0