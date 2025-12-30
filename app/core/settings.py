from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "CRITIQ-BIAS"
    ENV: str = "development"

    DATABASE_URL: str

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
