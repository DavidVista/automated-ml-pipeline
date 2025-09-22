FROM python:3.8-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY code/deployment/app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py
COPY code/deployment/app/app.py .

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
