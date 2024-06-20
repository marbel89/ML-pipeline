# Machine Learning Production Pipeline (WIP)

This project demonstrates how to put a machine learning model into production using an offline pipeline and creating an API. The key points covered include fetching data, running the model, saving predictions, and protecting credentials.

## Key Points (planned)

1. **Offline Pipeline Workflow**:
    - **Fetch Data**: Retrieve data from the database needed for model inference.
    - **Run Model**: Execute the model on the data (after cleaning and processing).
    - **Save Predictions**: Store the model predictions in a database table for easy retrieval using SQL queries.

2. **Command-Line Argument Parsing**:
    - `argparse` is a popular Python library for parsing command-line arguments when running Python scripts via a terminal.

3. **Model Serialization**:
    - `joblib` is a Python package that can handle saving and loading machine learning models.

4. **Credential Management**:
    - The `keyring` package can be used to protect credentials.

5. **API for Model Inference**:
    - An alternative way to put a model into production is to create an API.
    - APIs allow you to call a model residing on a different server or environment.

6. **REST API**:
    - One of the most common API architectures is REST.
    - REST APIs allow you to make HTTPS requests, such as GET or POST requests.

7. **FastAPI**:
    - FastAPI is a powerful library for creating APIs for your Python code.

## Getting Started

### Prerequisites

- Python 3.x
- `argparse` for command-line argument parsing
- `joblib` for model serialization
- `keyring` for credential management
- `FastAPI` for creating APIs
- `sklearn` for machine learning models

## Usage

1. **Offline Pipeline**:
    - Run the script to fetch data, run the model, and save predictions:
    ```bash
    python ml_pipeline_train_model.py -c config.json
    ```

2. **API**:
    - Create an API using FastAPI:
    ```bash
    uvicorn api:app --reload
    ```

## Project Structure
# ML Pipeline Project

This project demonstrates how to implement a machine learning pipeline for customer churn prediction. It includes scripts for data processing, model training, offline predictions, and configuration management.

## Project Structure
```
ML_Pipeline/
│
├── data/
│ ├── customer_churn_data.csv
│ └── customers/
│
├── customerData.py
├── dataSet.py
├── main.py
├── ml_model.py
├── ml_offline_predictions.py
├── ml_offline_predictions_config.json
├── ml_pipeline_config.json
├── ml_pipeline_train_model.py
└── README.md
```


