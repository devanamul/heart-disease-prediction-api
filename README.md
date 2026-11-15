# Heart Disease Prediction API

A machine learning REST API built with FastAPI to predict the presence of heart disease using Logistic Regression.

## Live Demo

- **API:** https://heart-disease-prediction-api-5pve.onrender.com
- **Swagger UI:** https://heart-disease-prediction-api-5pve.onrender.com/docs

## Technologies

- Python 3.11
- FastAPI
- Scikit-learn
- Pandas
- Pydantic
- Joblib
- Docker & Docker Compose
- Render

## Project Structure

    app/
      main.py
      schemas.py
    model/
      heart_model.joblib
    dataset/
      heart.csv
    train_model.py
    requirements.txt
    Dockerfile
    docker-compose.yml
    README.md

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /info | Model information |
| POST | /predict | Heart disease prediction |

## Run with Docker

    docker compose build
    docker compose up

Open: http://localhost:8000/docs

## Dataset

[Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
