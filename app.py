from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List
import asyncpg
from pathlib import Path
import logging
import json
from contextlib import asynccontextmanager

from solar_model_mgmt.config import DATABASE_URL, MODEL_STORAGE_PATH, AVAILABLE_FEATURES
from solar_model_mgmt.models import (
    UploadSuccessResponse,
    ModelResponse,
    ActiveModelResponse,
    PowerPlantDetailResponse,
    PowerPlantResponse,
    ModelsListResponse,
    ModelUpdateRequest,
    UpdateSuccessResponse,
    DeleteSuccessResponse,
    AvailableFeaturesResponse,
)
from solar_model_mgmt.database import (
    check_plant_exists,
    check_duplicate_model,
    store_model_file,
    insert_model_metadata,
    delete_model_file,
    cleanup_empty_directories,
    delete_model_metadata,
)
from solar_model_mgmt.validation import validate_file_extension, validate_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=1, max_size=10, command_timeout=60
        )
        logger.info("Database connection pool created successfully!")
    except Exception as e:
        logger.error(f"Database connection pool creation failed: {str(e)}")

    yield

    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


app = FastAPI(title="Solar Model Management API", version="1.0.0", lifespan=lifespan)


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    return {"message": "Solar Model Management API"}


@app.get("/features", response_model=AvailableFeaturesResponse)
async def get_available_features():
    """
    Get all available features that can be used in model training and prediction.
    """
    return AvailableFeaturesResponse(features=AVAILABLE_FEATURES)


# ============================================================================
# MODEL ENDPOINTS
# ============================================================================

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

        await validate_features(features_list)

        ext = await validate_file_extension(file.filename)
        # Derive file_type from extension (remove the leading dot)
        file_type = ext[1:] if ext.startswith(".") else ext

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await check_plant_exists(conn, plant_id)
                await check_duplicate_model(conn, model_name, version)

                model_dir = (
                    Path(MODEL_STORAGE_PATH)
                    / str(plant_id)
                    / model_name
                    / f"v{version}"
                )
                model_path = model_dir / f"model{ext}"

                await store_model_file(file, model_path)
                model_id = await insert_model_metadata(
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


@app.put("/models/{model_id}", response_model=UpdateSuccessResponse)
async def update_model_features(model_id: int, update_data: ModelUpdateRequest):
    """
    Update the features and is_active status of a specific model by ID.
    Both the features and is_active fields can be modified.
    """
    # Validate features before database operations
    await validate_features(update_data.features)

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
                model_info = await delete_model_metadata(conn, model_id)

                # Convert to Path object, handling both relative and absolute paths
                model_path = Path(model_info["path"])
                if not model_path.is_absolute():
                    model_path = model_path.resolve()

                # Delete the model file - if this fails, transaction will be rolled back
                await delete_model_file(model_path)

                await cleanup_empty_directories(model_path)

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


# ============================================================================
# POWER PLANT ENDPOINTS
# ============================================================================


@app.get("/power_plant/{id}", response_model=PowerPlantDetailResponse)
async def get_power_plant(id: int):
    """
    Get a single power plant by ID with model count.
    """
    async with db_pool.acquire() as conn:
        try:
            query = """
                SELECT 
                    pp.id,
                    pp.name,
                    pp.latitude,
                    pp.longitude,
                    pp.capacity,
                    COUNT(mm.id) as model_count
                FROM power_plant_v2 pp
                LEFT JOIN model_metadata mm ON pp.id = mm.plant_id
                WHERE pp.id = $1
                GROUP BY pp.id, pp.name, pp.latitude, pp.longitude, pp.capacity
            """

            row = await conn.fetchrow(query, id)

            if not row:
                raise HTTPException(
                    status_code=404, detail=f"Power plant with ID {id} not found"
                )

            return PowerPlantDetailResponse(
                id=row["id"],
                name=row["name"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                capacity=row["capacity"],
                model_count=row["model_count"],
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch power plant")


# ============================================================================
# INTERNAL ENDPOINTS
# ============================================================================

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
