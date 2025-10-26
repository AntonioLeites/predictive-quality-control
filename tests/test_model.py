from src.data_generation import generate_synthetic_data
from src.model_training import train_model

def test_model_training():
    df = generate_synthetic_data(100)
    model, scaler = train_model(df)
    assert model is not None

