FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install keras==3.10.0

# Copy app files
COPY app.py .
COPY eye_disease_model.keras .
COPY model_oct_c8/ ./model_oct_c8/
COPY fundus_final/ ./fundus_final/

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
