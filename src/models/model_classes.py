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
    """Modele baseline base sur la popularite des items avec personnalisation."""

    def __init__(self) -> None:
        self.item_scores: Optional[np.ndarray] = None
        self.n_items: int = 0
        self.user_preferences: Optional[np.ndarray] = None  # User genre preferences
        self.item_features: Optional[np.ndarray] = None  # Item genre features

    def fit(self, interaction_matrix: csr_matrix, **kwargs: Any) -> "PopularityModel":
        self.n_items = interaction_matrix.shape[1]
        item_counts = np.array(interaction_matrix.sum(axis=0)).flatten()
        self.item_scores = item_counts / (item_counts.max() + 1e-8)
        
        # Store interaction matrix for user preference computation
        self._train_matrix = interaction_matrix
        return self

    def _compute_user_genre_preferences(
        self,
        user_idx: int,
        item_features: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Compute user's genre preferences based on highly-rated items."""
        if item_features is None or self._train_matrix is None:
            return None
        
        # Get items the user has interacted with and their ratings
        user_ratings = self._train_matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return None
        
        # Weight genres by rating value (higher rated = stronger preference)
        preferences = np.zeros(item_features.shape[1], dtype=np.float32)
        for item_idx in rated_items:
            rating = user_ratings[item_idx]
            # Boost genres from highly-rated items (rating > 3.5)
            weight = max(0, (rating - 2.5) / 2.5)  # 0-1 scale for ratings 2.5-5
            preferences += item_features[item_idx] * weight
        
        # Normalize
        norm = np.linalg.norm(preferences)
        if norm > 0:
            preferences = preferences / norm
        
        return preferences

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_items: Optional[Set[int]] = None,
        item_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.item_scores is None:
            raise ValueError("Le modele n'a pas ete entraine")
        
        # Start with popularity scores
        scores = self.item_scores.copy()
        
        # Add personalization based on user genre preferences
        if item_features is not None and len(item_features) > 0:
            user_prefs = self._compute_user_genre_preferences(user_idx, item_features)
            if user_prefs is not None:
                # Compute content-based scores
                norms = np.linalg.norm(item_features, axis=1)
                norms[norms == 0] = 1
                content_scores = item_features.dot(user_prefs) / norms
                
                # Hybrid: 60% popularity + 40% content-based personalization
                scores = 0.6 * scores + 0.4 * content_scores
        
        if filter_items:
            for item_idx in filter_items:
                if item_idx < len(scores):
                    scores[item_idx] = -np.inf
        
        top_indices = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_indices]
        return top_indices, top_scores

    def similar_items(
        self,
        item_idx: int,
        n: int = 10,
        item_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne des items similaires bases sur les features ou la popularite.
        
        Si item_features (matrice de features genre) est fournie, calcule
        la similarite cosinus. Sinon, retourne les items les plus populaires.
        
        Args:
            item_idx: Index de l'item de reference.
            n: Nombre d'items similaires a retourner.
            item_features: Matrice optionnelle de features des items (n_items, n_features).
            
        Returns:
            Tuple (indices des items similaires, scores de similarite).
        """
        if item_features is not None and len(item_features) > 0:
            # Calculer la similarite cosinus basee sur les features
            item_vector = item_features[item_idx]
            
            # Norme L2 de chaque item
            norms = np.linalg.norm(item_features, axis=1)
            norms[norms == 0] = 1  # Eviter division par zero
            
            # Similarite cosinus
            similarities = item_features.dot(item_vector) / (norms * np.linalg.norm(item_vector) + 1e-8)
            
            # Exclure l'item de reference
            similarities[item_idx] = -np.inf
            
            top_indices = np.argsort(similarities)[::-1][:n]
            top_scores = similarities[top_indices]
            return top_indices, top_scores
        
        # Fallback: retourner les items les plus populaires
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
