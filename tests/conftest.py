

import pytest
from starlette.config import environ
from starlette.testclient import TestClient

environ["API_KEY"] = "test_key"

from turkishdelightnlp.main import get_app  # noqa: E402


@pytest.fixture()
def test_client():
    app = get_app()
    with TestClient(app) as test_client:
        yield test_client
