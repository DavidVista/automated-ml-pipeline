FROM python:3.8-slim-bullseye

RUN pip install mlflow==2.17.2

WORKDIR /home/mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
