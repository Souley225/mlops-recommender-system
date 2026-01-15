# -*- coding: utf-8 -*-
"""
Module de generation de recommandations.

Ce module fournit des fonctions pour generer des recommandations
a partir d'un modele entraine, incluant le filtrage et le formatage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Import model classes FIRST so joblib can find them when deserializing
from src.models.model_classes import PopularityModel, ALSModel  # noqa: F401

from src.utils.io import load_csv, load_joblib, load_json, load_sparse_matrix
from src.utils.logging import get_logger
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)


class Recommender:
    """
    Classe de recommandation encapsulant le modele et les metadonnees.

    Cette classe fournit une interface unifiee pour generer des recommandations,
    gerant le mapping des IDs et le formatage des resultats.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        """
        Initialise le recommandeur.

        Args:
            model_path: Chemin vers le modele. Par defaut: models/model.joblib.
            data_path: Chemin vers les donnees. Par defaut: data/processed.
        """
        self.model_path = model_path or MODELS_DIR / "model.joblib"
        self.data_path = data_path or PROCESSED_DATA_DIR

        self.model: Optional[Any] = None
        self.user_encoder: Optional[Any] = None
        self.item_encoder: Optional[Any] = None
        self.train_matrix: Optional[csr_matrix] = None
        self.item_popularity: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.item_features: Optional[np.ndarray] = None  # Genre-based features for similarity

        self._loaded = False

    def load(self) -> "Recommender":
        """
        Charge le modele et les donnees necessaires.

        Returns:
            L'instance du recommandeur charge.
        """
        if self._loaded:
            return self

        logger.info("Chargement du recommandeur")

        # Charger le modele
        self.model = load_joblib(self.model_path)

        # Charger les encodeurs
        self.user_encoder = load_joblib(self.data_path / "user_encoder.joblib")
        self.item_encoder = load_joblib(self.data_path / "item_encoder.joblib")

        # Charger la matrice d'entrainement pour filtrer les items vus
        self.train_matrix = load_sparse_matrix(self.data_path / "train_matrix.npz")

        # Charger les metadonnees
        self.metadata = load_json(self.data_path / "feature_metadata.json")

        # Charger la popularite des items (optionnel)
        popularity_path = self.data_path / "item_popularity.csv"
        if popularity_path.exists():
            self.item_popularity = load_csv(popularity_path)

        # Charger les infos des films (optionnel)
        movies_path = self.data_path.parent / "interim" / "movies.csv"
        if movies_path.exists():
            self.movies_df = load_csv(movies_path)
            # Construire la matrice de features genre pour la similarite
            self._build_genre_features()

        self._loaded = True

        logger.info(
            "Recommandeur charge",
            n_users=len(self.user_encoder.classes_),
            n_items=len(self.item_encoder.classes_),
        )

        return self

    def _build_genre_features(self) -> None:
        """
        Construit une matrice de features basee sur les genres.
        
        Chaque film est represente par un vecteur binaire one-hot
        encodant ses genres. Utilise pour la similarite content-based.
        """
        if self.movies_df is None or self.item_encoder is None:
            return
        
        # Obtenir tous les genres uniques
        all_genres = set()
        for genres_str in self.movies_df["genres"].dropna():
            if genres_str and genres_str != "(no genres listed)":
                for genre in genres_str.split("|"):
                    all_genres.add(genre.strip())
        
        genre_list = sorted(list(all_genres))
        genre_to_idx = {g: i for i, g in enumerate(genre_list)}
        
        n_items = len(self.item_encoder.classes_)
        n_genres = len(genre_list)
        
        if n_genres == 0:
            logger.warning("Aucun genre trouve dans les donnees")
            return
        
        # Creer la matrice de features
        self.item_features = np.zeros((n_items, n_genres), dtype=np.float32)
        
        for item_idx, item_id in enumerate(self.item_encoder.classes_):
            movie = self.movies_df[self.movies_df["item_id"] == item_id]
            if not movie.empty:
                genres_str = movie.iloc[0].get("genres", "")
                if genres_str and genres_str != "(no genres listed)":
                    for genre in genres_str.split("|"):
                        genre = genre.strip()
                        if genre in genre_to_idx:
                            self.item_features[item_idx, genre_to_idx[genre]] = 1.0
        
        logger.debug(
            "Matrice de features genre construite",
            n_items=n_items,
            n_genres=n_genres,
        )

    def _ensure_loaded(self) -> None:
        """Verifie que le recommandeur est charge."""
        if not self._loaded:
            self.load()

    def get_user_idx(self, user_id: int) -> Optional[int]:
        """
        Convertit un ID utilisateur en index.

        Args:
            user_id: ID utilisateur original.

        Returns:
            Index encode, ou None si inconnu.
        """
        self._ensure_loaded()

        try:
            return int(self.user_encoder.transform([user_id])[0])
        except ValueError:
            return None

    def get_item_id(self, item_idx: int) -> int:
        """
        Convertit un index item en ID original.

        Args:
            item_idx: Index encode de l'item.

        Returns:
            ID original de l'item.
        """
        self._ensure_loaded()
        return int(self.item_encoder.classes_[item_idx])

    def get_item_idx(self, item_id: int) -> Optional[int]:
        """
        Convertit un ID item en index.

        Args:
            item_id: ID item original.

        Returns:
            Index encode, ou None si inconnu.
        """
        self._ensure_loaded()

        try:
            return int(self.item_encoder.transform([item_id])[0])
        except ValueError:
            return None

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Genere des recommandations pour un utilisateur.

        Args:
            user_id: ID de l'utilisateur.
            n: Nombre de recommandations.
            exclude_seen: Exclure les items deja vus.
            include_metadata: Inclure les metadonnees des items.

        Returns:
            Liste de dictionnaires avec les recommandations.

        Example:
            >>> recommender = Recommender().load()
            >>> recs = recommender.recommend(user_id=1, n=5)
            >>> recs[0]
            {'item_id': 123, 'score': 0.95, 'title': 'Movie Title', 'rank': 1}
        """
        self._ensure_loaded()

        # Convertir l'ID utilisateur
        user_idx = self.get_user_idx(user_id)

        if user_idx is None:
            logger.warning(
                "Utilisateur inconnu, utilisation des recommandations populaires",
                user_id=user_id,
            )
            return self.recommend_popular(n, include_metadata)

        # Generer les recommandations
        try:
            if exclude_seen and self.train_matrix is not None:
                # Try to pass item_features for hybrid scoring
                try:
                    item_indices, scores = self.model.recommend(
                        user_idx,
                        n=n,
                        filter_items=set(self.train_matrix[user_idx].indices),
                        item_features=self.item_features,
                    )
                except TypeError:
                    # Fallback if model doesn't accept item_features
                    item_indices, scores = self.model.recommend(
                        user_idx,
                        n=n,
                        filter_items=set(self.train_matrix[user_idx].indices),
                    )
            else:
                try:
                    item_indices, scores = self.model.recommend(
                        user_idx, n=n, item_features=self.item_features
                    )
                except TypeError:
                    item_indices, scores = self.model.recommend(user_idx, n=n)
        except TypeError:
            # Fallback si le modele ne supporte pas filter_items
            seen_items = set(self.train_matrix[user_idx].indices) if exclude_seen else set()
            item_indices, scores = self.model.recommend(
                user_idx, n=n, filter_items=seen_items
            )

        # Formater les resultats
        recommendations = []
        for rank, (item_idx, score) in enumerate(zip(item_indices, scores), 1):
            item_id = self.get_item_id(item_idx)
            rec = {
                "item_id": item_id,
                "score": float(score),
                "rank": rank,
            }

            if include_metadata and self.movies_df is not None:
                item_info = self.movies_df[self.movies_df["item_id"] == item_id]
                if not item_info.empty:
                    rec["title"] = item_info.iloc[0].get("title", f"Item {item_id}")
                    rec["genres"] = item_info.iloc[0].get("genres", "")
                else:
                    rec["title"] = f"Item {item_id}"
                    rec["genres"] = ""

            recommendations.append(rec)

        logger.debug(
            "Recommandations generees",
            user_id=user_id,
            n_recommendations=len(recommendations),
        )

        return recommendations

    def recommend_popular(
        self,
        n: int = 10,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les items les plus populaires.

        Utilise comme fallback pour les nouveaux utilisateurs (cold start).

        Args:
            n: Nombre de recommandations.
            include_metadata: Inclure les metadonnees des items.

        Returns:
            Liste des items les plus populaires.
        """
        self._ensure_loaded()

        if self.item_popularity is None:
            logger.warning("Donnees de popularite non disponibles")
            return []

        # Top-n items populaires
        top_items = self.item_popularity.nlargest(n, "popularity_score")

        recommendations = []
        for rank, (_, row) in enumerate(top_items.iterrows(), 1):
            item_id = int(row["item_id"])
            rec = {
                "item_id": item_id,
                "score": float(row["popularity_score"]),
                "rank": rank,
            }

            if include_metadata and self.movies_df is not None:
                item_info = self.movies_df[self.movies_df["item_id"] == item_id]
                if not item_info.empty:
                    rec["title"] = item_info.iloc[0].get("title", f"Item {item_id}")
                    rec["genres"] = item_info.iloc[0].get("genres", "")

            recommendations.append(rec)

        return recommendations

    def similar_items(
        self,
        item_id: int,
        n: int = 10,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Trouve les items similaires a un item donne.

        Args:
            item_id: ID de l'item de reference.
            n: Nombre d'items similaires.
            include_metadata: Inclure les metadonnees des items.

        Returns:
            Liste des items similaires.
        """
        self._ensure_loaded()

        # Convertir l'ID item
        item_idx = self.get_item_idx(item_id)

        if item_idx is None:
            logger.warning("Item inconnu", item_id=item_id)
            return []

        # Verifier que le modele supporte la similarite
        if not hasattr(self.model, "similar_items"):
            logger.warning("Le modele ne supporte pas la similarite d'items")
            return []

        # Calculer les similarites
        # Pour PopularityModel, passer les features genre pour le calcul de similarite
        if hasattr(self.model, "similar_items"):
            # Verifier si le modele accepte item_features (PopularityModel)
            import inspect
            sig = inspect.signature(self.model.similar_items)
            if "item_features" in sig.parameters:
                similar_indices, scores = self.model.similar_items(
                    item_idx, n=n, item_features=self.item_features
                )
            else:
                similar_indices, scores = self.model.similar_items(item_idx, n=n)
        else:
            logger.warning("Le modele ne supporte pas la similarite d'items")
            return []

        # Formater les resultats
        similar = []
        for rank, (sim_idx, score) in enumerate(zip(similar_indices, scores), 1):
            sim_id = self.get_item_id(sim_idx)
            item = {
                "item_id": sim_id,
                "similarity": float(score),
                "rank": rank,
            }

            if include_metadata and self.movies_df is not None:
                item_info = self.movies_df[self.movies_df["item_id"] == sim_id]
                if not item_info.empty:
                    item["title"] = item_info.iloc[0].get("title", f"Item {sim_id}")
                    item["genres"] = item_info.iloc[0].get("genres", "")

            similar.append(item)

        return similar

    def get_all_users(self) -> List[int]:
        """
        Retourne la liste de tous les IDs utilisateurs connus.

        Returns:
            Liste des IDs utilisateurs.
        """
        self._ensure_loaded()
        return list(map(int, self.user_encoder.classes_))

    def get_all_items(self) -> List[int]:
        """
        Retourne la liste de tous les IDs items connus.

        Returns:
            Liste des IDs items.
        """
        self._ensure_loaded()
        return list(map(int, self.item_encoder.classes_))

    def get_user_history(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Retourne l'historique d'interactions d'un utilisateur.

        Args:
            user_id: ID de l'utilisateur.

        Returns:
            Liste des items avec lesquels l'utilisateur a interagi.
        """
        self._ensure_loaded()

        user_idx = self.get_user_idx(user_id)
        if user_idx is None:
            return []

        # Items vus dans l'entrainement
        seen_indices = self.train_matrix[user_idx].indices
        seen_values = self.train_matrix[user_idx].data

        history = []
        for item_idx, value in zip(seen_indices, seen_values):
            item_id = self.get_item_id(item_idx)
            item = {
                "item_id": item_id,
                "rating": float(value),
            }

            if self.movies_df is not None:
                item_info = self.movies_df[self.movies_df["item_id"] == item_id]
                if not item_info.empty:
                    item["title"] = item_info.iloc[0].get("title", f"Item {item_id}")

            history.append(item)

        return history
