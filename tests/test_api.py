# -*- coding: utf-8 -*-
"""
Tests pour l'API FastAPI.

Ces tests verifient le bon fonctionnement des endpoints de l'API
de recommandation.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Mock le recommandeur avant d'importer l'API
@pytest.fixture
def mock_recommender():
    """Fixture pour mocker le recommandeur."""
    with patch("src.serving.api.Recommender") as mock_class:
        mock_instance = MagicMock()

        # Configuration des retours mock
        mock_instance.get_all_users.return_value = [1, 2, 3, 4, 5]
        mock_instance.get_all_items.return_value = [10, 20, 30, 40, 50]
        mock_instance.recommend.return_value = [
            {"item_id": 10, "score": 0.95, "rank": 1, "title": "Film A", "genres": "Action"},
            {"item_id": 20, "score": 0.85, "rank": 2, "title": "Film B", "genres": "Comedy"},
        ]
        mock_instance.similar_items.return_value = [
            {"item_id": 20, "similarity": 0.9, "rank": 1, "title": "Film B", "genres": "Comedy"},
        ]
        mock_instance.get_user_history.return_value = [
            {"item_id": 10, "rating": 4.5, "title": "Film A"},
        ]

        mock_class.return_value.load.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(mock_recommender):
    """Fixture pour le client de test."""
    # Patch le recommandeur global dans le module api
    with patch("src.serving.api.recommender", mock_recommender):
        from src.serving.api import app
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Tests pour le endpoint /health."""

    def test_health_returns_200(self, client) -> None:
        """Verifie que /health retourne 200."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_response_structure(self, client) -> None:
        """Verifie la structure de la reponse."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data


class TestRecommendEndpoint:
    """Tests pour le endpoint /recommend."""

    def test_recommend_returns_200(self, client) -> None:
        """Verifie que /recommend retourne 200."""
        response = client.post(
            "/recommend",
            json={"user_id": 1, "k": 5},
        )

        assert response.status_code == 200

    def test_recommend_response_structure(self, client) -> None:
        """Verifie la structure de la reponse."""
        response = client.post(
            "/recommend",
            json={"user_id": 1, "k": 5},
        )
        data = response.json()

        assert "user_id" in data
        assert "recommendations" in data
        assert "timestamp" in data
        assert isinstance(data["recommendations"], list)

    def test_recommend_item_structure(self, client) -> None:
        """Verifie la structure des items recommandes."""
        response = client.post(
            "/recommend",
            json={"user_id": 1, "k": 5},
        )
        data = response.json()

        if data["recommendations"]:
            item = data["recommendations"][0]
            assert "item_id" in item
            assert "score" in item
            assert "rank" in item

    def test_recommend_validates_k(self, client) -> None:
        """Verifie la validation du parametre k."""
        response = client.post(
            "/recommend",
            json={"user_id": 1, "k": 0},  # k invalide
        )

        assert response.status_code == 422  # Validation error


class TestSimilarItemsEndpoint:
    """Tests pour le endpoint /similar-items."""

    def test_similar_items_returns_200(self, client) -> None:
        """Verifie que /similar-items retourne 200."""
        response = client.post(
            "/similar-items",
            json={"item_id": 10, "k": 5},
        )

        assert response.status_code == 200

    def test_similar_items_response_structure(self, client) -> None:
        """Verifie la structure de la reponse."""
        response = client.post(
            "/similar-items",
            json={"item_id": 10, "k": 5},
        )
        data = response.json()

        assert "item_id" in data
        assert "similar_items" in data
        assert "timestamp" in data


class TestUsersEndpoint:
    """Tests pour le endpoint /users."""

    def test_users_returns_200(self, client) -> None:
        """Verifie que /users retourne 200."""
        response = client.get("/users")

        assert response.status_code == 200

    def test_users_response_structure(self, client) -> None:
        """Verifie la structure de la reponse."""
        response = client.get("/users")
        data = response.json()

        assert "users" in data
        assert "count" in data
        assert isinstance(data["users"], list)

    def test_users_pagination(self, client) -> None:
        """Verifie la pagination."""
        response = client.get("/users?limit=2&offset=1")
        data = response.json()

        assert len(data["users"]) <= 2


class TestItemsEndpoint:
    """Tests pour le endpoint /items."""

    def test_items_returns_200(self, client) -> None:
        """Verifie que /items retourne 200."""
        response = client.get("/items")

        assert response.status_code == 200

    def test_items_response_structure(self, client) -> None:
        """Verifie la structure de la reponse."""
        response = client.get("/items")
        data = response.json()

        assert "items" in data
        assert "count" in data
