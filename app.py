from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncpg
from pathlib import Path
import shutil
from config import AppConfig

ALLOWED_EXTENSIONS = {".joblib", ".pkl", ".sav", ".h5", ".pt", ".onnx"}

app = FastAPI()


def get_config() -> AppConfig:
    return AppConfig()


async def get_db_pool(config: AppConfig = Depends(get_config)):
    if not hasattr(app.state, "db_pool"):
        app.state.db_pool = await asyncpg.create_pool(
            host=config.postgres_host,
            port=config.postgres_port,
            database=config.postgres_db,
            user=config.postgres_user,
            password=config.postgres_password,
        )
    return app.state.db_pool


@app.post("/models/")
async def upload_model(
    file: UploadFile = File(...),
    plant_id: int = Form(...),
    model_name: str = Form(...),
    version: int = Form(...),
    type: str = Form(...),
    features: str = Form(...),
    config: AppConfig = Depends(get_config),
    db_pool=Depends(get_db_pool),
):
    ext = Path(file.filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File extension {ext} not allowed."
        )

    pool = db_pool
    async with pool.acquire() as conn:
        plant = await conn.fetchrow(
            "SELECT id FROM power_plant_v2 WHERE id = $1", plant_id
        )
        if not plant:
            raise HTTPException(status_code=404, detail="Plant ID does not exist.")
        duplicate = await conn.fetchrow(
            "SELECT id FROM model_metadata WHERE name = $1 AND version = $2",
            model_name,
            version,
        )
        if duplicate:
            raise HTTPException(
                status_code=409,
                detail="Model with this name and version already exists.",
            )

    model_dir = (
        Path(config.model_storage_path) / str(plant_id) / model_name / f"v{version}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model{ext}"
    if model_path.exists():
        raise HTTPException(status_code=409, detail="Model file already exists.")

    with model_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO model_metadata (name, type, version, path, features) VALUES ($1, $2, $3, $4, $5)",
            model_name,
            type,
            version,
            str(model_path),
            features,
        )

    return JSONResponse(
        status_code=201,
        content={"message": "Model uploaded and metadata stored successfully."},
    )
