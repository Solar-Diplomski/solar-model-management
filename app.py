from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import asyncpg
from pathlib import Path
import shutil
import os
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/solar"
)
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./models")

ALLOWED_EXTENSIONS = {".joblib", ".pkl", ".sav", ".h5", ".pt", ".onnx"}

db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        # Initialize connection pool
        db_pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=10, command_timeout=60
        )
        logger.info("Database connection pool created successfully!")
    except Exception as e:
        logger.error(f"Database connection pool creation failed: {str(e)}")

    yield

    # Cleanup: Close the connection pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


app = FastAPI(title="Solar Model Management API", version="1.0.0", lifespan=lifespan)


class UploadSuccessResponse(BaseModel):
    message: str
    model_id: int


async def _validate_file_extension(filename: str) -> str:
    """Validate that the uploaded file has an allowed extension."""
    ext = Path(filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File extension {ext} not allowed."
        )
    return ext


async def _check_plant_exists(conn: asyncpg.Connection, plant_id: int) -> None:
    """Check if the specified plant ID exists in the database."""
    plant = await conn.fetchrow("SELECT id FROM power_plant_v2 WHERE id = $1", plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant ID does not exist.")


async def _check_duplicate_model(
    conn: asyncpg.Connection, model_name: str, version: int
) -> None:
    """Check if a model with the same name and version already exists."""
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
    """Store the uploaded model file to the specified path."""
    if model_path.exists():
        raise HTTPException(status_code=409, detail="Model file already exists.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with model_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()


async def _insert_model_metadata(
    conn: asyncpg.Connection,
    model_name: str,
    model_type: str,
    version: int,
    model_path: Path,
    features: str,
    plant_id: int,
    is_active: bool,
    file_type: str,
) -> int:
    """Insert model metadata into the database and return the model ID."""
    result = await conn.fetchrow(
        """
        INSERT INTO model_metadata (name, type, version, path, features, plant_id, is_active, file_type) 
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8) 
        RETURNING id
        """,
        model_name,
        model_type,
        version,
        str(model_path),
        features,
        plant_id,
        is_active,
        file_type,
    )
    return result["id"]


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {"message": "Solar Model Management API"}


@app.post("/models", response_model=UploadSuccessResponse)
async def upload_model(
    file: UploadFile = File(...),
    plant_id: int = Form(...),
    model_name: str = Form(...),
    version: int = Form(...),
    model_type: str = Form(...),
    features: str = Form(...),
    is_active: bool = Form(...),
):
    """
    Upload a new model file and store its metadata.

    The model file will be stored in the file system and its metadata
    will be saved to the database. The features should be provided as
    a JSON string. The file_type will be automatically derived from
    the uploaded file extension.
    """
    try:
        ext = await _validate_file_extension(file.filename)
        # Derive file_type from extension (remove the leading dot)
        file_type = ext[1:] if ext.startswith(".") else ext

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await _check_plant_exists(conn, plant_id)
                await _check_duplicate_model(conn, model_name, version)

                model_dir = (
                    Path(MODEL_STORAGE_PATH)
                    / str(plant_id)
                    / model_name
                    / f"v{version}"
                )
                model_path = model_dir / f"model{ext}"

                await _store_model_file(file, model_path)
                model_id = await _insert_model_metadata(
                    conn,
                    model_name,
                    model_type,
                    version,
                    model_path,
                    features,
                    plant_id,
                    is_active,
                    file_type,
                )

                logger.info(
                    f"Model uploaded successfully: {model_name} v{version} (ID: {model_id})"
                )

                return UploadSuccessResponse(
                    message="Model uploaded and metadata stored successfully",
                    model_id=model_id,
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload model")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
