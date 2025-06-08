from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List
import asyncpg
from pathlib import Path
import logging
import json
from contextlib import asynccontextmanager
import httpx
from datetime import datetime, timedelta

from solar_model_mgmt.config import (
    DATABASE_URL,
    MODEL_STORAGE_PATH,
    AVAILABLE_FEATURES,
    PREDICTION_SERVICE_URL,
)
from solar_model_mgmt.models import (
    UploadSuccessResponse,
    ModelResponse,
    ActiveModelResponse,
    PowerPlantDetailResponse,
    PowerPlantResponse,
    PowerPlantOverviewResponse,
    PowerPlantCreateRequest,
    PowerPlantCreateResponse,
    PowerPlantUpdateRequest,
    PowerPlantUpdateResponse,
    ForecastResponse,
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


@app.get("/power_plant", response_model=List[PowerPlantDetailResponse])
async def get_power_plants():
    """
    Get all power plants with model count.
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
                GROUP BY pp.id, pp.name, pp.latitude, pp.longitude, pp.capacity
                ORDER BY pp.name
            """

            rows = await conn.fetch(query)

            return [
                PowerPlantDetailResponse(
                    id=row["id"],
                    name=row["name"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    capacity=row["capacity"],
                    model_count=row["model_count"],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch power plants")


@app.get("/power_plant/overview", response_model=List[PowerPlantOverviewResponse])
async def get_power_plants_overview():
    """
    Get overview for all power plants with forecasts from their first active model.
    """
    async with db_pool.acquire() as conn:
        try:
            # Get all power plants
            plant_query = """
                SELECT 
                    pp.id,
                    pp.name,
                    pp.latitude,
                    pp.longitude
                FROM power_plant_v2 pp
                ORDER BY pp.name
            """

            plant_rows = await conn.fetch(plant_query)

            # Get first active model for each plant
            model_query = """
                SELECT DISTINCT ON (mm.plant_id)
                    mm.plant_id,
                    mm.id,
                    mm.name
                FROM model_metadata mm
                WHERE mm.is_active = true
                ORDER BY mm.plant_id, mm.id
            """

            model_rows = await conn.fetch(model_query)

            # Create a map of plant_id to model info
            plant_models = {
                row["plant_id"]: {"id": row["id"], "name": row["name"]}
                for row in model_rows
            }

            # Calculate time period: 00:00:00 of current day to 00:00:00 of next day
            now = datetime.now()
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            overviews = []

            for plant_row in plant_rows:
                forecasts = []
                plant_id = plant_row["id"]

                if plant_id in plant_models:
                    model_info = plant_models[plant_id]
                    try:
                        async with httpx.AsyncClient() as client:
                            forecast_url = f"{PREDICTION_SERVICE_URL}/internal/forecast/{model_info['id']}"
                            params = {
                                "start_date": start_date_str,
                                "end_date": end_date_str,
                            }

                            response = await client.get(forecast_url, params=params)

                            if response.status_code == 200:
                                forecast_data = response.json()

                                # Transform the forecast data to include model name
                                forecasts = [
                                    ForecastResponse(
                                        id=model_info["id"],
                                        name=model_info["name"],
                                        prediction_time=item["prediction_time"],
                                        power_output=item["power_output"],
                                    )
                                    for item in forecast_data
                                ]

                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch forecasts for plant {plant_id} from prediction service: {str(e)}"
                        )

                coordinates = []
                if (
                    plant_row["latitude"] is not None
                    and plant_row["longitude"] is not None
                ):
                    coordinates = [plant_row["latitude"], plant_row["longitude"]]

                overviews.append(
                    PowerPlantOverviewResponse(
                        id=plant_row["id"],
                        name=plant_row["name"],
                        forecasts=forecasts,
                        coordinates=coordinates,
                    )
                )

            return overviews

        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to fetch power plant overviews"
            )


@app.get("/power_plant/{plant_id}", response_model=PowerPlantDetailResponse)
async def get_power_plant(plant_id: int):
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

            row = await conn.fetchrow(query, plant_id)

            if not row:
                raise HTTPException(
                    status_code=404, detail=f"Power plant with ID {plant_id} not found"
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


@app.post("/power_plant", response_model=PowerPlantCreateResponse)
async def create_power_plant(plant_data: PowerPlantCreateRequest):
    """
    Create a new power plant.

    Creates a new power plant with the provided name, longitude, latitude, and capacity.
    All fields except name are optional.
    """
    async with db_pool.acquire() as conn:
        try:
            insert_query = """
                INSERT INTO power_plant_v2 (name, longitude, latitude, capacity)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
            """

            plant_id = await conn.fetchval(
                insert_query,
                plant_data.name,
                plant_data.longitude,
                plant_data.latitude,
                plant_data.capacity,
            )

            if plant_id is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Power plant with name '{plant_data.name}' already exists",
                )

            logger.info(
                f"Power plant created successfully: {plant_data.name} (ID: {plant_id})"
            )

            return PowerPlantCreateResponse(
                message="Power plant created successfully",
                power_plant_id=plant_id,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Power plant creation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create power plant")


@app.put("/power_plant/{plant_id}", response_model=PowerPlantUpdateResponse)
async def update_power_plant(plant_id: int, update_data: PowerPlantUpdateRequest):
    """
    Update an existing power plant.

    Updates the power plant with the provided data. All fields are optional.
    Only provided fields will be updated. At least one field must be provided.
    """
    update_fields = {k: v for k, v in update_data.model_dump().items() if v is not None}

    if not update_fields:
        raise HTTPException(
            status_code=400, detail="At least one field must be provided for update"
        )

    async with db_pool.acquire() as conn:
        try:
            check_query = """
                SELECT id, name 
                FROM power_plant_v2 
                WHERE id = $1
            """

            existing_plant = await conn.fetchrow(check_query, plant_id)

            if not existing_plant:
                raise HTTPException(
                    status_code=404, detail=f"Power plant with ID {plant_id} not found"
                )

            # If name is being updated, check for conflicts
            if (
                "name" in update_fields
                and update_fields["name"] != existing_plant["name"]
            ):
                name_check_query = """
                    SELECT id 
                    FROM power_plant_v2 
                    WHERE name = $1 AND id != $2
                """

                existing_name = await conn.fetchrow(
                    name_check_query, update_fields["name"], plant_id
                )

                if existing_name:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Power plant with name '{update_fields['name']}' already exists",
                    )

            set_clauses = []
            values = []
            param_idx = 1

            for field, value in update_fields.items():
                set_clauses.append(f"{field} = ${param_idx}")
                values.append(value)
                param_idx += 1

            values.append(plant_id)

            update_query = f"""
                UPDATE power_plant_v2 
                SET {", ".join(set_clauses)}
                WHERE id = ${param_idx}
            """

            await conn.execute(update_query, *values)

            logger.info(
                f"Power plant updated successfully: ID {plant_id} with fields {list(update_fields.keys())}"
            )

            return PowerPlantUpdateResponse(
                message="Power plant updated successfully",
                power_plant_id=plant_id,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Power plant update failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update power plant")


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
