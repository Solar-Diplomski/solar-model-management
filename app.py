from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import List, Any
import asyncpg
from pathlib import Path
import shutil
import os
import logging
import json
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


class ModelResponse(BaseModel):
    id: int
    name: str
    type: str
    version: int
    features: Any
    plant_name: str
    is_active: bool
    file_type: str

    @field_validator("features")
    @classmethod
    def parse_features(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


class ActiveModelResponse(BaseModel):
    id: int
    features: Any
    plant_id: int
    file_type: str

    @field_validator("features")
    @classmethod
    def parse_features(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v


class PowerPlantResponse(BaseModel):
    id: int
    longitude: float | None
    latitude: float | None
    capacity: float | None


class ModelsListResponse(BaseModel):
    models: List[ModelResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class ModelUpdateRequest(BaseModel):
    features: List[str]
    is_active: bool


class UpdateSuccessResponse(BaseModel):
    message: str
    model_id: int


class DeleteSuccessResponse(BaseModel):
    message: str


class AvailableFeaturesResponse(BaseModel):
    features: List[str]


available_features = [
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


async def _validate_file_extension(filename: str) -> str:
    """Validate that the uploaded file has an allowed extension."""
    ext = Path(filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File extension {ext} not allowed."
        )
    return ext


async def _validate_features(features: List[str]) -> None:
    """Validate that all provided features are in the available features list."""
    invalid_features = [f for f in features if f not in available_features]
    if invalid_features:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid features provided: {invalid_features}. "
            f"Available features: {available_features}",
        )


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


async def _delete_model_file(model_path: Path) -> None:
    """Delete the model file from the filesystem."""
    if not model_path.exists():
        logger.warning(f"Model file not found for deletion: {model_path}")
        return

    try:
        model_path.unlink()
        logger.info(f"Model file deleted successfully: {model_path}")
    except Exception as e:
        logger.error(f"Failed to delete model file {model_path}: {str(e)}")
        raise HTTPException(status_code=500)


async def _cleanup_empty_directories(model_path: Path) -> None:
    """Clean up empty parent directories after model file deletion."""
    try:
        # Start from the model file's parent directory and work up
        current_dir = model_path.parent

        # Stop at the MODEL_STORAGE_PATH to avoid deleting the root storage directory
        storage_path = Path(MODEL_STORAGE_PATH).resolve()

        while current_dir != storage_path and current_dir.parent != current_dir:
            try:
                current_dir.rmdir()
                logger.info(f"Removed empty directory: {current_dir}")
                current_dir = current_dir.parent
            except OSError:
                # Directory not empty or cannot be removed, stop cleanup
                break
    except Exception as e:
        logger.warning(
            f"Failed to cleanup empty directories for {model_path}: {str(e)}"
        )


async def _delete_model_metadata(conn: asyncpg.Connection, model_id: int) -> dict:
    """Delete model metadata from the database and return the deleted model info."""
    model_info = await conn.fetchrow(
        "SELECT name, version, path, plant_id FROM model_metadata WHERE id = $1",
        model_id,
    )

    if not model_info:
        raise HTTPException(
            status_code=404, detail=f"Model with ID {model_id} not found"
        )

    await conn.execute("DELETE FROM model_metadata WHERE id = $1", model_id)

    return dict(model_info)


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {"message": "Solar Model Management API"}


@app.get("/features", response_model=AvailableFeaturesResponse)
async def get_available_features():
    """
    Get all available features that can be used in model training and prediction.
    """
    return AvailableFeaturesResponse(features=available_features)


@app.get("/models", response_model=ModelsListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
):
    """
    Get a paginated list of all models from the model_metadata table.
    Results are sorted by plant name, then by model name, then by version.
    """
    offset = (page - 1) * page_size

    async with db_pool.acquire() as conn:
        try:
            # Query to get total count
            count_query = """
                SELECT COUNT(*) 
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
            """
            total_count = await conn.fetchval(count_query)

            query = """
                SELECT 
                    mm.id,
                    mm.name,
                    mm.type,
                    mm.version,
                    mm.features,
                    mm.is_active,
                    mm.file_type,
                    pp.name as plant_name
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
                ORDER BY pp.name, mm.name, mm.version
                LIMIT $1 OFFSET $2
            """

            rows = await conn.fetch(query, page_size, offset)

            models = [
                ModelResponse(
                    id=row["id"],
                    name=row["name"],
                    type=row["type"],
                    version=row["version"],
                    features=row["features"],
                    plant_name=row["plant_name"],
                    is_active=row["is_active"],
                    file_type=row["file_type"],
                )
                for row in rows
            ]

            total_pages = (total_count + page_size - 1) // page_size

            return ModelsListResponse(
                models=models,
                total_count=total_count,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
            )

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch models")


@app.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int):
    """
    Get a single model by ID from the model_metadata table.
    """
    async with db_pool.acquire() as conn:
        try:
            query = """
                SELECT 
                    mm.id,
                    mm.name,
                    mm.type,
                    mm.version,
                    mm.features,
                    mm.is_active,
                    mm.file_type,
                    pp.name as plant_name
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
                WHERE mm.id = $1
            """

            row = await conn.fetchrow(query, model_id)

            if not row:
                raise HTTPException(
                    status_code=404, detail=f"Model with ID {model_id} not found"
                )

            return ModelResponse(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                version=row["version"],
                features=row["features"],
                plant_name=row["plant_name"],
                is_active=row["is_active"],
                file_type=row["file_type"],
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch model")


@app.put("/models/{model_id}", response_model=UpdateSuccessResponse)
async def update_model_features(model_id: int, update_data: ModelUpdateRequest):
    """
    Update the features and is_active status of a specific model by ID.
    Both the features and is_active fields can be modified.
    """
    # Validate features before database operations
    await _validate_features(update_data.features)

    async with db_pool.acquire() as conn:
        try:
            check_query = """
                SELECT 
                    mm.id,
                    mm.name,
                    mm.type,
                    mm.version,
                    mm.features,
                    mm.is_active,
                    mm.file_type,
                    pp.name as plant_name
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
                WHERE mm.id = $1
            """

            existing_model = await conn.fetchrow(check_query, model_id)

            if not existing_model:
                raise HTTPException(
                    status_code=404, detail=f"Model with ID {model_id} not found"
                )

            update_query = """
                UPDATE model_metadata 
                SET features = $1, is_active = $2
                WHERE id = $3
            """

            await conn.execute(
                update_query,
                json.dumps(update_data.features),
                update_data.is_active,
                model_id,
            )

            return UpdateSuccessResponse(
                message="Model features and status updated successfully",
                model_id=model_id,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database update failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to update model features"
            )


@app.delete("/models/{model_id}", response_model=DeleteSuccessResponse)
async def delete_model(model_id: int):
    """
    Delete a specific model by ID.
    Removes both the database record and the model file from the filesystem.
    If file deletion fails, the database transaction is rolled back.
    """
    async with db_pool.acquire() as conn:
        try:
            async with conn.transaction():
                model_info = await _delete_model_metadata(conn, model_id)

                # Convert to Path object, handling both relative and absolute paths
                model_path = Path(model_info["path"])
                if not model_path.is_absolute():
                    model_path = model_path.resolve()

                # Delete the model file - if this fails, transaction will be rolled back
                await _delete_model_file(model_path)

                await _cleanup_empty_directories(model_path)

                logger.info(
                    f"Model deleted successfully: {model_info['name']} v{model_info['version']} "
                    f"(ID: {model_id}, Plant: {model_info['plant_id']})"
                )

                return DeleteSuccessResponse(message="Model deleted successfully")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Model deletion failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete model")


@app.get(
    "/internal/power-plant/active",
    response_model=List[PowerPlantResponse],
)
async def get_power_plants_with_active_models():
    """
    Internal endpoint that returns all power plants that have at least one active model.
    Returns the power plant ID, longitude, and latitude for each plant.
    """
    async with db_pool.acquire() as conn:
        try:
            query = """
                SELECT DISTINCT
                    pp.id,
                    pp.longitude,
                    pp.latitude,
                    pp.capacity
                FROM power_plant_v2 pp
                INNER JOIN model_metadata mm ON pp.id = mm.plant_id
                WHERE mm.is_active = true
                ORDER BY pp.id
            """

            rows = await conn.fetch(query)

            power_plants = [
                PowerPlantResponse(
                    id=row["id"],
                    longitude=row["longitude"],
                    latitude=row["latitude"],
                    capacity=row["capacity"],
                )
                for row in rows
            ]

            return power_plants

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch power plants with active models",
            )


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
        # Parse and validate features
        try:
            features_list = json.loads(features)
            if not isinstance(features_list, list):
                raise HTTPException(
                    status_code=400, detail="Features must be a JSON array of strings"
                )
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Features must be valid JSON")

        await _validate_features(features_list)

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


@app.get("/internal/models/active", response_model=List[ActiveModelResponse])
async def get_active_models():
    """
    Get all active models from the model_metadata table.
    Results are sorted by plant name, then by model name, then by version.
    """
    async with db_pool.acquire() as conn:
        try:
            query = """
                SELECT 
                    mm.id,
                    mm.features,
                    mm.file_type,
                    mm.plant_id
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
                WHERE mm.is_active = true
                ORDER BY pp.name, mm.name, mm.version
            """

            rows = await conn.fetch(query)

            models = [
                ActiveModelResponse(
                    id=row["id"],
                    features=row["features"],
                    plant_id=row["plant_id"],
                    file_type=row["file_type"],
                )
                for row in rows
            ]

            return models

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch active models")


@app.get("/internal/models/{model_id}/download")
async def download_model(model_id: int):
    """
    Download the model file for a specific model ID.
    Returns the actual model file as a downloadable attachment.
    """
    async with db_pool.acquire() as conn:
        try:
            query = """
                SELECT 
                    mm.id,
                    mm.name,
                    mm.version,
                    mm.path,
                    mm.file_type,
                    mm.plant_id,
                    pp.name as plant_name
                FROM model_metadata mm
                JOIN power_plant_v2 pp ON mm.plant_id = pp.id
                WHERE mm.id = $1
            """

            row = await conn.fetchrow(query, model_id)

            if not row:
                raise HTTPException(
                    status_code=404, detail=f"Model with ID {model_id} not found"
                )

            model_path = Path(row["path"])

            # Handle relative paths stored in database (like "models\1\rf\v5\model.joblib")
            if not model_path.is_absolute():
                model_path = model_path.resolve()

            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found on disk: {model_path}",
                )

            filename = f"{row['plant_name']}_{row['name']}_v{row['version']}.{row['file_type']}"

            return FileResponse(
                path=str(model_path),
                filename=filename,
                media_type="application/octet-stream",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Model download failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to download model")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
