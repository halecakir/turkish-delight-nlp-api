

from typing import Callable

from fastapi import FastAPI
from loguru import logger

from turkishdelightnlp.core.config import DEFAULT_MODELS
from turkishdelightnlp.services.models import load_models


def _startup_model(app: FastAPI) -> None:
    models_info = DEFAULT_MODELS
    model_instances = load_models(models_info)
    app.state.model = model_instances


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)
    return shutdown
