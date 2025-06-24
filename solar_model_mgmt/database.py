import asyncpg
import logging
from fastapi import HTTPException, UploadFile
from pathlib import Path
import shutil
from .config import MODEL_STORAGE_PATH

logger = logging.getLogger(__name__)


async def check_plant_exists(conn: asyncpg.Connection, plant_id: int) -> None:
    """Check if the specified plant ID exists in the database."""
    plant = await conn.fetchrow("SELECT id FROM power_plant WHERE id = $1", plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant ID does not exist.")


async def check_duplicate_model(
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


async def store_model_file(file: UploadFile, model_path: Path) -> None:
    """Store the uploaded model file to the specified path."""
    if model_path.exists():
        raise HTTPException(status_code=409, detail="Model file already exists.")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with model_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()


async def insert_model_metadata(
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


async def delete_model_file(model_path: Path) -> None:
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


async def cleanup_empty_directories(model_path: Path) -> None:
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


async def delete_model_metadata(conn: asyncpg.Connection, model_id: int) -> dict:
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
