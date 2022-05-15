

from starlette.config import Config
from starlette.datastructures import Secret
import srsly

APP_VERSION = "0.1"
APP_NAME = "Turkish Delight NLP API"
API_PREFIX = "/api"

config = Config("config/dev.env")

API_KEY: Secret = config("API_KEY", cast=Secret)
IS_DEBUG: bool = config("IS_DEBUG", cast=bool, default=False)

DEFAULT_MODELS: dict = srsly.read_json(config("DEFAULT_MODELS"))
