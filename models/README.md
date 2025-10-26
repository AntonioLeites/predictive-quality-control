Trained models will be saved here

uvicorn src.prediction_api:app --reload

----
## 1️⃣ uvicorn
Uvicorn is a lightweight ASGI server for Python.

ASGI (Asynchronous Server Gateway Interface) is the standard for running FastAPI, Starlette, and other async web apps.

Essentially, uvicorn takes your FastAPI application and serves it over HTTP so you can access it in your browser or via API calls.

## 2️⃣ src.prediction_api:app
This part has two components:

src.prediction_api → Python module path

Tells uvicorn where your code lives. In your project:

css
Copy code
predictive-quality-control/
└── src/
    └── prediction_api.py
app → the FastAPI instance inside that module

In prediction_api.py, you probably have:

python
Copy code
from fastapi import FastAPI
app = FastAPI()
uvicorn uses this app object to know what to run.

## 3️⃣ --reload
Enables hot reload during development.

Every time you save a change in your code, the server automatically reloads.

Great for development because you don’t need to restart uvicorn manually.

⚠️ Not recommended for production—use it only in dev environments.

## ✅ Summary

The command starts a development server for your FastAPI app.

You can then call endpoints like:

http://127.0.0.1:8000/predict


--reload lets you see code changes instantly.