# -*- coding: utf-8 -*-
"""
Tests pour le module de features.

Ces tests verifient le bon fonctionnement des fonctions de construction
des features: matrices, normalisation, popularite.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.features.build_features import (
    binarize_ratings,
    build_confidence_matrix,
    build_interaction_matrix,
    compute_popularity,
    normalize_ratings,
)


class TestBuildInteractionMatrix:
    """Tests pour la construction de la matrice d'interactions."""

    def test_creates_correct_shape(self) -> None:
        """Verifie que la matrice a la bonne forme."""
        df = pd.DataFrame({
            "user_idx": [0, 1, 2],
            "item_idx": [0, 1, 2],
            "rating": [4.0, 3.0, 5.0],
        })

        matrix = build_interaction_matrix(df, n_users=3, n_items=3)

        assert matrix.shape == (3, 3)

    def test_contains_correct_values(self) -> None:
        """Verifie que les valeurs sont correctes."""
        df = pd.DataFrame({
            "user_idx": [0, 1],
            "item_idx": [0, 1],
            "rating": [4.0, 3.0],
        })

        matrix = build_interaction_matrix(df, n_users=2, n_items=2)

        assert matrix[0, 0] == 4.0
        assert matrix[1, 1] == 3.0
        assert matrix[0, 1] == 0.0  # Pas d'interaction

    def test_handles_sparse_data(self) -> None:
        """Verifie le support des donnees sparse."""
        df = pd.DataFrame({
            "user_idx": [0, 99],
            "item_idx": [0, 99],
            "rating": [4.0, 3.0],
        })

        matrix = build_interaction_matrix(df, n_users=100, n_items=100)

        assert matrix.shape == (100, 100)
        assert matrix.nnz == 2


class TestNormalizeRatings:
    """Tests pour la normalisation des notes."""

    def test_mean_centering(self) -> None:
        """Verifie le centrage par la moyenne."""
        df = pd.DataFrame({
            "user_id": [1, 1, 1],
            "rating": [3.0, 4.0, 5.0],  # Moyenne = 4.0
        })

        result, metadata = normalize_ratings(df, method="mean_centering")

        assert "rating_normalized" in result.columns
        # Les notes centrees doivent avoir une moyenne proche de 0
        assert abs(result["rating_normalized"].mean()) < 0.01

    def test_min_max_normalization(self) -> None:
        """Verifie la normalisation min-max."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "rating": [1.0, 3.0, 5.0],
        })

        result, metadata = normalize_ratings(df, method="min_max")

        assert result["rating_normalized"].min() == 0.0
        assert result["rating_normalized"].max() == 1.0


class TestComputePopularity:
    """Tests pour le calcul de la popularite."""

    def test_popularity_scores_range(self) -> None:
        """Verifie que les scores sont entre 0 et 1."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 1, 2],
            "item_id": [1, 1, 1, 2, 2],
            "rating": [4.0, 5.0, 3.0, 4.0, 4.0],
            "timestamp": [100, 200, 300, 400, 500],
        })

        result = compute_popularity(df, time_decay=False)

        assert result["popularity_score"].min() >= 0.0
        assert result["popularity_score"].max() <= 1.0

    def test_more_interactions_higher_popularity(self) -> None:
        """Verifie que plus d'interactions = plus de popularite."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 1],
            "item_id": [1, 1, 1, 2],  # Item 1 a 3 interactions, item 2 en a 1
            "rating": [4.0, 4.0, 4.0, 4.0],
            "timestamp": [100, 200, 300, 400],
        })

        result = compute_popularity(df, time_decay=False)

        item1_pop = result[result["item_id"] == 1]["popularity_score"].values[0]
        item2_pop = result[result["item_id"] == 2]["popularity_score"].values[0]

        assert item1_pop > item2_pop


class TestBinarizeRatings:
    """Tests pour la binarisation des notes."""

    def test_default_threshold(self) -> None:
        """Verifie le seuil par defaut."""
        df = pd.DataFrame({
            "rating": [1.0, 2.0, 3.0, 3.5, 4.0, 5.0],
        })

        result = binarize_ratings(df, threshold=3.5)

        expected = [0, 0, 0, 1, 1, 1]
        assert list(result["interaction"].astype(int)) == expected

    def test_returns_float32(self) -> None:
        """Verifie le type de donnees."""
        df = pd.DataFrame({"rating": [4.0, 5.0]})
        result = binarize_ratings(df)

        assert result["interaction"].dtype == np.float32


class TestBuildConfidenceMatrix:
    """Tests pour la matrice de confiance."""

    def test_confidence_increases_with_alpha(self) -> None:
        """Verifie que la confiance augmente avec alpha."""
        df = pd.DataFrame({
            "user_idx": [0, 1],
            "item_idx": [0, 1],
            "interaction": [1.0, 1.0],
        })

        matrix_low = build_confidence_matrix(df, alpha=10)
        matrix_high = build_confidence_matrix(df, alpha=100)

        assert matrix_high[0, 0] > matrix_low[0, 0]

    def test_minimum_confidence_is_one(self) -> None:
        """Verifie que la confiance minimale est 1."""
        df = pd.DataFrame({
            "user_idx": [0],
            "item_idx": [0],
            "interaction": [0.0],  # Interaction nulle
        })

        matrix = build_confidence_matrix(df, alpha=40)

        # Confiance = 1 + alpha * 0 = 1
        assert matrix[0, 0] == 1.0
