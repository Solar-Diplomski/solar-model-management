from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    model_storage_path: str = "./models"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "solar"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    class Config:
        env_file = ".env.dev"
