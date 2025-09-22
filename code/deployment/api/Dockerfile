# FastAPI Dockerfile
FROM python:3.8-slim-bullseye

# Set the working directory
WORKDIR /api

# Copy and install requirements
COPY code/deployment/api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy api.py and model files
COPY code/deployment/api/api.py .

# Run the FastAPI server
CMD ["uvicorn", "api:api", "--host", "0.0.0.0", "--port", "8000"]
