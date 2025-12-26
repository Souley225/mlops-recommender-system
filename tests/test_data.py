# -*- coding: utf-8 -*-
"""
Tests pour le module de donnees.

Ces tests verifient le bon fonctionnement des fonctions de telechargement,
de creation et de division du dataset.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import (
    clean_ratings,
    convert_to_implicit,
    create_interactions_dataset,
)
from src.data.split_dataset import (
    create_encoders,
    encode_ids,
    filter_cold_start,
    temporal_split,
)


class TestCleanRatings:
    """Tests pour la fonction clean_ratings."""

    def test_removes_invalid_ratings(self) -> None:
        """Verifie que les notes invalides sont supprimees."""
        df = pd.DataFrame({
            "userId": [1, 2, 3, 4],
            "movieId": [1, 2, 3, 4],
            "rating": [4.0, 0.0, 6.0, 3.5],  # 0.0 et 6.0 sont invalides
            "timestamp": [100, 200, 300, 400],
        })

        result = clean_ratings(df, min_rating=0.5, max_rating=5.0)

        assert len(result) == 2
        assert all(result["rating"] >= 0.5)
        assert all(result["rating"] <= 5.0)

    def test_removes_duplicates_keeps_last(self) -> None:
        """Verifie que les doublons sont supprimes en gardant le dernier."""
        df = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [1, 1, 2],
            "rating": [3.0, 4.0, 5.0],
            "timestamp": [100, 200, 300],
        })

        result = clean_ratings(df)

        assert len(result) == 2
        # La note 4.0 (timestamp 200) doit etre gardee
        assert result[result["userId"] == 1]["rating"].values[0] == 4.0


class TestConvertToImplicit:
    """Tests pour la conversion en feedback implicite."""

    def test_binarizes_ratings(self) -> None:
        """Verifie la binarisation des notes."""
        df = pd.DataFrame({
            "rating": [1.0, 2.5, 3.5, 4.0, 5.0],
        })

        result = convert_to_implicit(df, rating_threshold=3.5)

        assert "interaction" in result.columns
        assert list(result["interaction"]) == [0, 0, 1, 1, 1]

    def test_custom_threshold(self) -> None:
        """Verifie le seuil personnalise."""
        df = pd.DataFrame({
            "rating": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = convert_to_implicit(df, rating_threshold=4.0)

        assert list(result["interaction"]) == [0, 0, 0, 1, 1]


class TestFilterColdStart:
    """Tests pour le filtrage cold-start."""

    def test_filters_users_with_few_interactions(self) -> None:
        """Verifie le filtrage des utilisateurs avec peu d'interactions."""
        df = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            "item_id": [1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5],
        })

        result = filter_cold_start(df, min_user_interactions=5, min_item_interactions=1)

        # Utilisateur 2 doit etre filtre (seulement 2 interactions)
        assert 2 not in result["user_id"].values

    def test_filters_items_with_few_interactions(self) -> None:
        """Verifie le filtrage des items avec peu d'interactions."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "item_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        })

        # Item 1 et 2 ont 5 interactions chacun, donc tout passe
        result = filter_cold_start(df, min_user_interactions=1, min_item_interactions=5)

        assert len(result) == 10


class TestTemporalSplit:
    """Tests pour la division temporelle."""

    def test_split_ratios(self) -> None:
        """Verifie les ratios de division."""
        df = pd.DataFrame({
            "user_id": range(100),
            "item_id": range(100),
            "rating": [4.0] * 100,
            "timestamp": range(100),
        })

        train, val, test = temporal_split(df, 0.8, 0.1, 0.1)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_temporal_order_preserved(self) -> None:
        """Verifie que l'ordre temporel est preserve."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [1, 2, 3],
            "rating": [4.0, 4.0, 4.0],
            "timestamp": [100, 200, 300],
        })

        train, val, test = temporal_split(df, 0.34, 0.33, 0.33)

        # Les timestamps doivent etre croissants entre les splits
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()


class TestEncoders:
    """Tests pour les encodeurs."""

    def test_create_encoders(self) -> None:
        """Verifie la creation des encodeurs."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 1, 2],
            "item_id": [10, 20, 30, 20, 30],
        })

        user_encoder, item_encoder = create_encoders(df)

        assert len(user_encoder.classes_) == 3
        assert len(item_encoder.classes_) == 3

    def test_encode_ids(self) -> None:
        """Verifie l'encodage des IDs."""
        train_df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
        })

        user_encoder, item_encoder = create_encoders(train_df)

        test_df = pd.DataFrame({
            "user_id": [1, 2],
            "item_id": [10, 20],
        })

        result = encode_ids(test_df, user_encoder, item_encoder)

        assert "user_idx" in result.columns
        assert "item_idx" in result.columns
        assert all(result["user_idx"] >= 0)
        assert all(result["item_idx"] >= 0)
