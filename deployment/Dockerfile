FROM python:3.10-slim
WORKDIR /app

COPY deployment/requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "src.prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]

