FROM apache/airflow:2.7.0

# Install Docker
RUN pip install docker python-dotenv

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/code"
ENV SEED="42"

# Create directory structure
RUN mkdir -p /opt/airflow/code/datasets \
             /opt/airflow/code/deployment \
             /opt/airflow/code/models \
             /opt/airflow/data/raw \
             /opt/airflow/data/processed \
             /opt/airflow/models

# Copy code
COPY code/ /opt/airflow/code/
COPY services/airflow/dags/ /opt/airflow/dags/
COPY data/raw /opt/airflow/data/raw
