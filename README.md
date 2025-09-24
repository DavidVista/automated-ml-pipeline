# automated-ml-pipeline
A clear and multi-functional pipeline to facilitate deployment of a model. The pipeline has multiple stages including data engineering, training of a model, and deployment of the app. All contents are containerized with Docker

## Structure

**Apache Airflow** is a powerful framework for orchestrating multiple stages in a ML pipeline. In this project, a pipeline is automated with data processing, model engineering, and model deployment stages as parts inside Docker containers. Execution plan consists of the following parts:
1. Setting seed number.
2. Data processing (loading, imputation, scaling, and splitting).
3. Model engineering (training, hyperparameter setup, logging metrics, and dumping model).
4. Model deployment (starting api and app containers).

The first stage is used to set seed number for different python packages.

The data processing stage loads data in `data/raw` connected as a folder volume and creates a dataset (train and test sets) inside a folder volume `data/processed`.

To facilitate monitoring of model performance, **MLFlow** framework is employed in the model engineering stage. A separate container is run for handling logging request. This webserver also provides an informative IU that combines all metrics and model information in one place.

After the model is trained, airflow dumps it insider a *named docker volume* (`automated-ml-pipeline_models`). This mechanism ensures that api and app containers have access to the saved model.

Api and app are run at the last stage as separate containers on the host machine. Implementation is possible with the use of docker socket. The airflow worker has docker installed inside its container, and its socket is replaced with the socket of the docker on the user machine using a directory volume.

The pipeline has a timer mechanism that restarts the procedure after 10 minutes.

## Comments on Runs
Note that the first run may take 5-7 minutes due to setup of docker builder. Later stages usually take 2-3 minutes to run. An exception can be due to problems with network access to docker image repository (TLS handshake error). The pipeline is set to have a retry after 30 seconds from a failure.

## Starting all up
1. Use docker compose to build and run airflow containers:
```bash
docker compose up -d
```

2. Wait for initialization of webclient and follow the address http://0.0.0.0:8080 (use default credentials login: `airflow`, password: `airflow`).

3. Start `ml_data_processing_pipeline` DAG.

4. Upon successful completion, you may follow different parts of the app:
- API: http://0.0.0.0:8000 (No UI);
- App frontend: http://0.0.0.0:8501;
- MLFlow UI: http://0.0.0.0:5000.

5. Finally, shutting Airflow and MLFlow containers down can be made by using this command:

```bash
docker compose down
```

Since api and app containers are run independently, shut down them as follows:
```bash
docker stop ml-app
docker stop ml-api
```
