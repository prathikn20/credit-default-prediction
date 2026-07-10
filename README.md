# Credit Default Prediction

This project is a train-and-serve pipeline for the UCI credit card default dataset, hosted publicly.

**Live demo:** https://credit-default-prediction-9u4i.onrender.com/docs

*Note: the web hosting takes about a minute to cold start and spin up the model.*

## Overview

I originally just had an exploratory notebook where I was trying different things with the data and models. I then came back to it, fixed a test-set leakage I identified in the way I was tuning hyperparameters, serialized, served, containerized, and then deployed the model.

The actual model itself isn't that impressive because the UCI dataset is widely regarded as a beginner dataset; it was my way of exploring this field. I was researching how I might be able to optimize the dataset, but due to the quality of the dataset the AUC caps around 0.78 despite the methods used. The value of this project is in the pipeline and deployment, where it's end to end and other people have access to the trained model. This experience will be valuable when I work with a more complicated dataset and train a model on it, since I'll know the process to deploy it.

## The Problem

The two major flaws I had to address in my exploratory model before I could deploy were, first, the test-set leakage. When tuning the hyperparameters I was checking the model's metrics against the test data, so I was optimizing the model for the test data. When deploying I made sure to have a train-validate-test split instead. The second flaw was a model/metric disconnect, where I serialized a bare `RandomForestClassifier` and reported metrics on another object that was trained.

## Methodology

- The model that was shipped works on a three-way (15/15/70) stratified split. Test was split off first, with 15% of the data, so that it wouldn't be used until basically the very end to report the model metrics. Then the remaining 85% was split so that 15% of the original dataset would be validation data for hyperparameter tuning. The model was then trained on the train + validation data.

- `n_estimators` was set to 200 because more than that wouldn't have contributed much to the model due to diminishing returns. I picked 200 because it was an appropriate number for the dataset, and I didn't add it to the hyperparameter tuning since I was using a manual grid search and one more parameter would've added more compute time. The rest of the parameters were chosen by widening the grid until the results stopped changing and explicitly plateaued, rather than assuming it.

- I also used `class_weight='balanced'` since the dataset was imbalanced.

- I originally had `get_dummies` in my exploratory notebook to choose the column order, but then I swapped it for `OneHotEncoder` since it was stateful. `get_dummies` doesn't learn anything, so when serving the model a single incoming row would produce different columns than training, while `OneHotEncoder` remembers training categories and handles unseen categories, making it the most reasonable choice when deploying.

## Results

Test AUC 0.781, Validation AUC 0.785, 0.75 recall, 0.37 precision at a 0.4 threshold. Both metrics landing this close to each other is evidence that the leakage is gone, since if it had persisted the test AUC should have been lower.

I chose the threshold at 0.4 instead of the default to optimize recall. I swept between 0.3, 0.4, and 0.5. The reason to optimize recall is that from a business perspective a missed default costs more than a false alarm. The reason I chose 0.4 specifically is that it's an optimal elbow, where there's reasonable precision for the recall value. Past 0.4, the collapse in precision outweighs the gain in recall.

## Model Serving

A FastAPI app exposes a `POST /predict` endpoint. The app loads `credit_pipeline.joblib` from disk so that it's read a single time and reused. Input validation is a Pydantic schema with 23 fields, one typed field per feature, ensuring that malformed input never reaches the model.

The 0.4 threshold lives in the serving layer, not the joblib, so that it can move without needing retraining.

The inputted JSON is turned into a single-row DataFrame and given to the model, so that the pipeline's stateful encoder aligns with it.

Consistent output, with classification and probability returned regardless of the decision. Verified by running a real row through `/docs` and checking that the probability was properly returned and the label matched the threshold.

Auto docs at `/docs` (a FastAPI test page built from the schema). Run locally with:

```
python -m uvicorn src.app:app --reload
```

from the repo root.

One thing to acknowledge is that although the page loads and the model works consistently, that doesn't mean the model is correct.

## Model Deployment

The model needs to be containerized since a joblib model is version-sensitive, so I used Docker to freeze the different versions of modules and Python, the dependencies, code, and model into a reproducible image.

`scikit-learn` is pinned despite not being imported in `app.py`, since joblib rebuilds sklearn objects when unpickling.

The model is deployed on Render and the port is read from `${PORT:-8000}`. I originally changed the Render language to Python, which caused the deployment to fail since it needed to be Docker. Then deployment failed again since my joblib model was in my `.gitignore` and I forgot to force-add it, but after doing that the model successfully deployed.

There was a change in the hosted model's output where the last digit was off by 1. The containerized prediction matches local to 15 significant figures, so the difference is floating-point non-determinism across CPUs.

## Stack

Python, pandas, scikit-learn, joblib for training; FastAPI, uvicorn, pydantic for serving; Docker; Render.

## Running it Locally

From the repo root (paths resolve from there):

- Retrain the model:

```
python src/train.py
```

- Serve locally:

```
python -m uvicorn src.app:app --reload
```

- Build and run the container:

```
docker build -t credit-default .
docker run -p 8000:8000 credit-default
```