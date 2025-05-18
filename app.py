from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import asyncpg
from pathlib import Path
import shutil
from config import AppConfig
import asyncio
from contextlib import asynccontextmanager

ALLOWED_EXTENSIONS = {".joblib", ".pkl", ".sav", ".h5", ".pt", ".onnx"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    db_pool = getattr(app.state, "db_pool", None)
    if db_pool is not None:
        try:
            await db_pool.close()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


def get_config() -> AppConfig:
    return AppConfig()


def get_db_pool_lock():
    if not hasattr(app.state, "db_pool_lock"):
        app.state.db_pool_lock = asyncio.Lock()
    return app.state.db_pool_lock


async def get_db_pool(config: AppConfig = Depends(get_config)):
    lock = get_db_pool_lock()
    async with lock:
        if not hasattr(app.state, "db_pool"):
            app.state.db_pool = await asyncpg.create_pool(
                host=config.postgres_host,
                port=config.postgres_port,
                database=config.postgres_db,
                user=config.postgres_user,
                password=config.postgres_password,
                min_size=1,
                max_size=10,
            )
    return app.state.db_pool


async def _validate_file_extension(filename: str) -> str:
    ext = Path(filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File extension {ext} not allowed."
        )
    return ext


async def _check_plant_exists(conn: asyncpg.Connection, plant_id: int) -> None:
    plant = await conn.fetchrow("SELECT id FROM power_plant_v2 WHERE id = $1", plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant ID does not exist.")


async def _check_duplicate_model(
    conn: asyncpg.Connection, model_name: str, version: int
) -> None:
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


async def _store_model_file(file: UploadFile, model_path: Path) -> None:
    if model_path.exists():
        raise HTTPException(status_code=409, detail="Model file already exists.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()


async def _insert_model_metadata(
    conn: asyncpg.Connection,
    model_name: str,
    model_type: str,
    version: int,
    model_path: Path,
    features: str,
    plant_id: int,
) -> None:
    await conn.execute(
        "INSERT INTO model_metadata (name, type, version, path, features, plant_id) VALUES ($1, $2, $3, $4, $5, $6)",
        model_name,
        model_type,
        version,
        str(model_path),
        features,
        plant_id,
    )


@app.post("/models/")
async def upload_model(
    file: UploadFile = File(...),
    plant_id: int = Form(...),
    model_name: str = Form(...),
    version: int = Form(...),
    model_type: str = Form(...),
    features: str = Form(...),
    config: AppConfig = Depends(get_config),
    db_pool=Depends(get_db_pool),
):
    ext = await _validate_file_extension(file.filename)
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await _check_plant_exists(conn, plant_id)
            await _check_duplicate_model(conn, model_name, version)
            model_dir = (
                Path(config.model_storage_path)
                / str(plant_id)
                / model_name
                / f"v{version}"
            )
            model_path = model_dir / f"model{ext}"
            await _store_model_file(file, model_path)
            await _insert_model_metadata(
                conn, model_name, model_type, version, model_path, features, plant_id
            )
    return JSONResponse(
        status_code=201,
        content={"message": "Model uploaded and metadata stored successfully."},
    )
