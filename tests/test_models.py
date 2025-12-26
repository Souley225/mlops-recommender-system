# -*- coding: utf-8 -*-
"""
Tests pour le module de modeles.

Ces tests verifient le bon fonctionnement des algorithmes de recommandation
et des fonctions d'evaluation.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.train import ALSModel, PopularityModel
from src.utils.metrics import (
    compute_all_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


class TestPopularityModel:
    """Tests pour le modele de popularite."""

    def test_fit_computes_scores(self) -> None:
        """Verifie que fit calcule les scores de popularite."""
        # Matrice avec item 0 plus populaire que item 1
        matrix = csr_matrix(np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]))

        model = PopularityModel()
        model.fit(matrix)

        assert model.item_scores is not None
        assert model.item_scores[0] > model.item_scores[1]

    def test_recommend_returns_top_items(self) -> None:
        """Verifie que recommend retourne les items les plus populaires."""
        matrix = csr_matrix(np.array([
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.5],
        ]))

        model = PopularityModel()
        model.fit(matrix)

        indices, scores = model.recommend(user_idx=0, n=2)

        # Item 0 devrait etre en premier (somme = 3.0)
        assert indices[0] == 0
        assert len(indices) == 2

    def test_recommend_filters_items(self) -> None:
        """Verifie le filtrage des items."""
        matrix = csr_matrix(np.array([
            [1.0, 0.5, 0.3],
        ]))

        model = PopularityModel()
        model.fit(matrix)

        indices, scores = model.recommend(user_idx=0, n=2, filter_items={0})

        # Item 0 ne devrait pas etre dans les resultats
        assert 0 not in indices


class TestALSModel:
    """Tests pour le modele ALS."""

    def test_fit_creates_factors(self) -> None:
        """Verifie que fit cree les facteurs latents."""
        matrix = csr_matrix(np.random.rand(10, 20))

        model = ALSModel(factors=8, iterations=5)
        model.fit(matrix)

        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.user_factors.shape == (10, 8)
        assert model.item_factors.shape == (20, 8)

    def test_recommend_returns_correct_count(self) -> None:
        """Verifie que recommend retourne le bon nombre d'items."""
        matrix = csr_matrix(np.random.rand(10, 20))

        model = ALSModel(factors=8, iterations=5)
        model.fit(matrix)

        indices, scores = model.recommend(user_idx=0, n=5)

        assert len(indices) == 5
        assert len(scores) == 5

    def test_similar_items_excludes_self(self) -> None:
        """Verifie que similar_items exclut l'item lui-meme."""
        matrix = csr_matrix(np.random.rand(10, 20))

        model = ALSModel(factors=8, iterations=5)
        model.fit(matrix)

        indices, scores = model.similar_items(item_idx=5, n=5)

        # L'item 5 ne devrait pas etre dans les resultats
        assert 5 not in indices

    def test_get_params_returns_dict(self) -> None:
        """Verifie que get_params retourne les parametres."""
        model = ALSModel(factors=32, regularization=0.05, iterations=10)

        params = model.get_params()

        assert params["factors"] == 32
        assert params["regularization"] == 0.05
        assert params["iterations"] == 10


class TestMetrics:
    """Tests pour les metriques d'evaluation."""

    def test_rmse_calculation(self) -> None:
        """Verifie le calcul de RMSE."""
        y_true = np.array([4.0, 3.0, 5.0, 2.0])
        y_pred = np.array([3.5, 3.2, 4.8, 2.5])

        result = rmse(y_true, y_pred)

        assert result > 0
        assert result < 1.0  # Predictions proches

    def test_rmse_perfect_prediction(self) -> None:
        """Verifie RMSE avec prediction parfaite."""
        y_true = np.array([4.0, 3.0, 5.0])
        y_pred = np.array([4.0, 3.0, 5.0])

        result = rmse(y_true, y_pred)

        assert result == 0.0

    def test_precision_at_k(self) -> None:
        """Verifie le calcul de Precision@K."""
        relevant = {1, 2, 3}
        recommended = [1, 4, 2, 5, 3]

        p_at_5 = precision_at_k(relevant, recommended, k=5)
        p_at_3 = precision_at_k(relevant, recommended, k=3)

        assert p_at_5 == 3 / 5  # 3 pertinents sur 5
        assert p_at_3 == 2 / 3  # 2 pertinents sur 3

    def test_recall_at_k(self) -> None:
        """Verifie le calcul de Recall@K."""
        relevant = {1, 2, 3, 4, 5}
        recommended = [1, 6, 2, 7, 3]

        r_at_5 = recall_at_k(relevant, recommended, k=5)

        assert r_at_5 == 3 / 5  # 3 trouves sur 5 pertinents

    def test_ndcg_at_k_perfect_ranking(self) -> None:
        """Verifie NDCG avec un ranking parfait."""
        relevant = {1, 2, 3}
        recommended = [1, 2, 3, 4, 5]  # Ordre parfait

        ndcg = ndcg_at_k(relevant, recommended, k=5)

        assert ndcg == 1.0

    def test_ndcg_at_k_empty_relevant(self) -> None:
        """Verifie NDCG avec aucun item pertinent."""
        relevant: set = set()
        recommended = [1, 2, 3]

        ndcg = ndcg_at_k(relevant, recommended, k=3)

        assert ndcg == 0.0

    def test_compute_all_metrics(self) -> None:
        """Verifie le calcul de toutes les metriques."""
        all_relevant = [{1, 2}, {3, 4}]
        all_recommended = [[1, 3, 5], [3, 5, 4]]

        metrics = compute_all_metrics(all_relevant, all_recommended, k_values=[3])

        assert "precision_at_3" in metrics
        assert "recall_at_3" in metrics
        assert "ndcg_at_3" in metrics
        assert "mrr" in metrics
