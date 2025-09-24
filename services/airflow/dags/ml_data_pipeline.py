from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import random

import os

from datasets.preprocess import load_dataset, undersample, preprocess
from models.feature_engineering import transform
from models.training import GradientBoostingModel


MODEL_NAME = "gradient_boosting_classifier"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(seconds=30),
}


def set_seed_task(**kwargs):
    SEED = int(os.environ.get('SEED', '42'))
    np.random.seed(seed=SEED)
    random.seed(SEED)


def process_data_task(**kwargs):
    """Task to load, preprocess, and save data"""

    try:
        # Define paths
        raw_data_path = '/opt/airflow/data/raw/cs-training.csv'
        processed_train_path = '/opt/airflow/data/processed/train_dataset.csv'
        processed_test_path = '/opt/airflow/data/processed/test_dataset.csv'
        models_dir = '/opt/airflow/models'

        # Ensure directories exist
        os.makedirs('/opt/airflow/data/processed', exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # I. Load and process data
        X, y = load_dataset(raw_data_path)

        # Undersample
        X, y = undersample(X, y)

        # II. Split data
        SEED = int(os.environ.get('SEED', '42'))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )

        # III. Preprocess data
        X_train_processed = preprocess(X_train.values, train=True, models_dir=models_dir)

        X_test_processed = preprocess(X_test.values, train=False, models_dir=models_dir)

        # IV. Save processed data

        # Training data
        train_df = pd.DataFrame(X_train_processed, columns=X.columns)
        train_df['SeriousDlqin2yrs'] = y_train.values
        train_df.to_csv(processed_train_path, index=True)

        # Test data
        test_df = pd.DataFrame(X_test_processed, columns=X.columns)
        test_df['SeriousDlqin2yrs'] = y_test.values
        test_df.to_csv(processed_test_path, index=True)

        return {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_path': processed_train_path,
            'test_path': processed_test_path,
            'status': 'success'
        }

    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        print("Full traceback:")
        raise e


def model_engineering_task(**kwargs):
    try:
        # Define paths
        processed_train_path = '/opt/airflow/data/processed/train_dataset.csv'
        processed_test_path = '/opt/airflow/data/processed/test_dataset.csv'
        models_dir = '/opt/airflow/models'

        # Ensure directories exist
        os.makedirs('/opt/airflow/data/processed', exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # I. Load train and test data
        X_train, y_train = load_dataset(processed_train_path)
        X_test, y_test = load_dataset(processed_test_path)

        # II. Transform train data
        X_train = transform(X_train)
        X_test = transform(X_test)

        # III. Train model

        # Split into train and validation sets
        SEED = int(os.environ.get('SEED', '42'))

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )

        # Parameters
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'subsample': 1.0,
            'max_depth': 4,
            'n_iter_no_change': 10,
            'validation_fraction': 0.2
        }

        # Initialize model
        model = GradientBoostingModel(
            name=MODEL_NAME,
            model=GradientBoostingClassifier,
            X_train=X_train.values,
            **params
        )

        # Train model
        run_id = model.train(X_train, y_train)

        # Tune hyperparameter
        optimal_threshold = model.find_optimal_threshold(X_val, y_val, run_id)

        # Evaluate on test set using optimal threshold
        metrics = model.evaluate(X_test, y_test, run_id, use_optimal_threshold=True)

        # Save model
        model.dump(run_id, '/opt/airflow/models')

        return {
            'model_name': model.name,
            'optimal_threshold': optimal_threshold,
            'test_metrics': metrics,
            'test_samples': len(X_test),
            'status': 'success'
        }

    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        print("Full traceback:")
        raise e


with DAG(
    'ml_data_processing_pipeline',
    default_args=default_args,
    description='ML Automated Pipeline - Data engineering, Model engineering, Deployment',
    schedule_interval=timedelta(minutes=10),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'data-processing'],
) as dag:

    set_seed = PythonOperator(
        task_id='set_seed',
        python_callable=set_seed_task,
        dag=dag
    )

    process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data_task,
        provide_context=True,
        dag=dag,
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=model_engineering_task,
        provide_context=True,
        dag=dag
    )

    # Cleanup existing containers first
    cleanup_containers = BashOperator(
        task_id='cleanup_existing_containers',
        bash_command='''\
        # Stop and remove API container if exists
        docker stop ml-api || true
        docker rm ml-api || true

        # Stop and remove App container if exists
        docker stop ml-app || true
        docker rm ml-app || true
        ''',
        env={'DOCKER_HOST': 'unix:///var/run/docker.sock'},
    )

    # Build Docker images
    build_api_image = BashOperator(
        task_id='build_api_image',
        bash_command='cd /opt/airflow && docker build -f code/deployment/api/Dockerfile -t ml-api:latest .',
        env={
            'DOCKER_HOST': 'unix:///var/run/docker.sock',
            'API_IMAGE': 'ml-app:latest'
        },
    )

    build_app_image = BashOperator(
        task_id='build_app_image',
        bash_command='cd /opt/airflow && docker build -f code/deployment/app/Dockerfile -t ml-app:latest .',
        env={
            'DOCKER_HOST': 'unix:///var/run/docker.sock',
            'API_IMAGE': 'ml-app:latest'
        },
    )

    # Create a network for containers
    deploy_network = BashOperator(
        task_id='create_network',
        bash_command='docker network inspect app-network >/dev/null 2>&1 || docker network create app-network',
        env={'DOCKER_HOST': 'unix:///var/run/docker.sock'},
    )

    deploy_api = BashOperator(
        task_id='deploy_api',
        bash_command=f'''\
        docker run -d --name ml-api \
            --network app-network \
            -p 8000:8000 \
            --env MODEL_PATH=/app/models \
            --env MODEL_NAME={MODEL_NAME} \
            --expose 8000 \
            -v automated-ml-pipeline_models:/app/models \
            ml-api:latest
        ''',
        env={'DOCKER_HOST': 'unix:///var/run/docker.sock'},
    )

    deploy_app = BashOperator(
        task_id='deploy_app',
        bash_command='''\
        docker run -d --name ml-app \
            --network app-network \
            -p 8501:8501 \
            --env API_URL=http://ml-api:8000 \
            ml-app:latest
        ''',
        env={'DOCKER_HOST': 'unix:///var/run/docker.sock'},
    )

    set_seed >> process_data >> train_model \
        >> cleanup_containers \
        >> build_api_image >> build_app_image \
        >> deploy_network >> deploy_api >> deploy_app
