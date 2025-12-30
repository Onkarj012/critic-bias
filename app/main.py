from fastapi import FastAPI
from app.core.settings import settings
from app.core.logging import setup_logging

def create_app() -> FastAPI:
    setup_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.0.1"
    )

    return app

app = create_app()


