from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
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
