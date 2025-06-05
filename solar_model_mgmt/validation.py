from fastapi import HTTPException
from pathlib import Path
from typing import List
from .config import ALLOWED_EXTENSIONS, AVAILABLE_FEATURES


async def validate_file_extension(filename: str) -> str:
    """Validate that the uploaded file has an allowed extension."""
    ext = Path(filename).suffix
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"File extension {ext} not allowed."
        )
    return ext


async def validate_features(features: List[str]) -> None:
    """Validate that all provided features are in the available features list."""
    invalid_features = [f for f in features if f not in AVAILABLE_FEATURES]
    if invalid_features:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid features provided: {invalid_features}. "
            f"Available features: {AVAILABLE_FEATURES}",
        )
