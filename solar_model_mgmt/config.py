import os

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/solar"
)

# Prediction service configuration
PREDICTION_SERVICE_URL = os.getenv("PREDICTION_SERVICE_URL", "http://localhost:8001")

# Storage configuration
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./models")

# File validation
ALLOWED_EXTENSIONS = {".joblib", ".pkl", ".sav", ".h5", ".pt", ".onnx"}

# Available features for model training and prediction
AVAILABLE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "shortwave_radiation_instant",
    "diffuse_radiation",
    "diffuse_radiation_instant",
    "direct_normal_irradiance",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "is_day",
    "sunshine_duration",
    "hour",
    "month",
    "day",
    "day_of_year",
    "week_of_year",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "capacity",
    "power_plant_capacity",
    "latitude",
    "longitude",
    "elevation",
]
