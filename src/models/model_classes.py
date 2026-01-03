# -*- coding: utf-8 -*-
"""
Classes de modeles pour la deserialisation joblib.

Ce module contient les definitions minimales des classes de modeles
necessaires pour charger les modeles sauvegardes avec joblib.
"""

from typing import Any, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix


class PopularityModel:
    """Modele baseline base sur la popularite des items."""

    def __init__(self) -> None:
        self.item_scores: Optional[np.ndarray] = None
        self.n_items: int = 0

    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> "PopularityModel":
        self.n_items = interaction_matrix.shape[1]
        item_counts = np.array(interaction_matrix.sum(axis=0)).flatten()
        self.item_scores = item_counts / item_counts.max()
        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_items: Optional[Set[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.item_scores is None:
            raise ValueError("Le modele n'a pas ete entraine")
        scores = self.item_scores.copy()
        if filter_items:
            for item_idx in filter_items:
                if item_idx < len(scores):
                    scores[item_idx] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_indices]
        return top_indices, top_scores

    def similar_items(self, item_idx: int, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne les items les plus populaires comme fallback.
        
        Le modele de popularite n'a pas de notion de similarite entre items,
        donc on retourne les items les plus populaires (hors l'item de reference).
        """
        if self.item_scores is None:
            raise ValueError("Le modele n'a pas ete entraine")
        scores = self.item_scores.copy()
        # Exclure l'item de reference
        scores[item_idx] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_indices]
        return top_indices, top_scores


class ALSModel:
    """Modele ALS pour feedback implicite."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_items: Optional[Set[int]] = None,
        interaction_matrix: Optional[csr_matrix] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Le modele n'a pas ete entraine")
        user_vector = self.user_factors[user_idx]
        scores = self.item_factors.dot(user_vector)
        if filter_items:
            for item_idx in filter_items:
                if item_idx < len(scores):
                    scores[item_idx] = -np.inf
        if interaction_matrix is not None:
            seen_items = interaction_matrix[user_idx].indices
            scores[seen_items] = -np.inf
        top_indices = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_indices]
        return top_indices, top_scores

    def similar_items(self, item_idx: int, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.item_factors is None:
            raise ValueError("Le modele n'a pas ete entraine")
        item_vector = self.item_factors[item_idx]
        similarities = self.item_factors.dot(item_vector)
        similarities[item_idx] = -np.inf
        top_indices = np.argsort(similarities)[::-1][:n]
        top_scores = similarities[top_indices]
        return top_indices, top_scores

    def get_params(self) -> dict:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "use_gpu": self.use_gpu,
        }
