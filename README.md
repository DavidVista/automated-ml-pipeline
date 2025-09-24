# automated-ml-pipeline
A clear and multi-functional pipeline to facilitate deployment of a model. The pipeline has multiple stages including data engineering, training of a model, and deployment of the app. All contents are containerized with Docker

## Starting all up
1. Use docker compose to build and run airflow containers:
```bash
docker compose up -d
```

2. Wait for initialization of webclient and follow the address http://0.0.0.0:8080.

3. Start `ml_data_processing_pipeline` DAG

4. Upon successful completion, you may follow different parts of the app:
- API: http://0.0.0.0:8000;
- App frontend: http://0.0.0.0:8501;
- MLFlow UI: http://0.0.0.0:5000.
