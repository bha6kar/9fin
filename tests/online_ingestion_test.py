import pytest
from src.online_ingestion_api import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_get_item_id_handler_valid(client):
    """Test the /api/<doc_id> endpoint with a valid document ID."""
    response = client.get("/api/007")
    assert response.status_code == 200


def test_get_item_id_handler_invalid(client):
    """Test the /api/<doc_id> endpoint with an invalid document ID."""
    response = client.get("/api/123")
    assert response.status_code == 404
