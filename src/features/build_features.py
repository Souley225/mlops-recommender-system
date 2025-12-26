# -*- coding: utf-8 -*-
"""
Module de construction des features pour le systeme de recommandation.

Ce module transforme les interactions brutes en features utilisables
par les modeles de recommandation, incluant:
- Construction de matrices sparse user-item
- Normalisation des notes
- Calcul des popularites
- Preparation des features pour differents types de modeles
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse import csr_matrix

from src.utils.io import (
    load_joblib,
    load_parquet,
    save_json,
    save_parquet,
    save_sparse_matrix,
)
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

logger = get_logger(__name__)


def build_interaction_matrix(
    df: pd.DataFrame,
    n_users: int,
    n_items: int,
    value_col: str = "rating",
) -> csr_matrix:
    """
    Construit une matrice sparse d'interactions user-item.

    Cette fonction cree une matrice CSR (Compressed Sparse Row) efficace
    pour stocker les interactions, optimisee pour les operations de lecture
    par ligne (acces aux interactions d'un utilisateur).

    Args:
        df: DataFrame avec colonnes user_idx, item_idx, et value_col.
        n_users: Nombre total d'utilisateurs.
        n_items: Nombre total d'items.
        value_col: Nom de la colonne contenant les valeurs (rating ou interaction).

    Returns:
        Matrice sparse CSR de forme (n_users, n_items).

    Example:
        >>> matrix = build_interaction_matrix(train_df, n_users=1000, n_items=5000)
        >>> matrix.shape
        (1000, 5000)
    """
    rows = df["user_idx"].values
    cols = df["item_idx"].values
    values = df[value_col].values

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    logger.info(
        "Matrice d'interactions construite",
        shape=matrix.shape,
        nnz=matrix.nnz,
        density=round(matrix.nnz / (n_users * n_items) * 100, 4),
    )

    return matrix


def normalize_ratings(
    df: pd.DataFrame,
    method: str = "mean_centering",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalise les notes dans le DataFrame.

    La normalisation aide les modeles a mieux generaliser en eliminant
    les biais utilisateurs (certains utilisateurs notent toujours haut).

    Args:
        df: DataFrame avec colonne 'rating'.
        method: Methode de normalisation:
            - 'mean_centering': Centre autour de la moyenne utilisateur
            - 'z_score': Normalisation z-score par utilisateur
            - 'min_max': Mise a l'echelle [0, 1] globale

    Returns:
        Tuple (DataFrame normalise, metadata de normalisation).

    Example:
        >>> normalized_df, meta = normalize_ratings(train_df, method='mean_centering')
    """
    df = df.copy()
    metadata: Dict[str, Any] = {"method": method}

    if method == "mean_centering":
        # Calculer la moyenne par utilisateur
        user_means = df.groupby("user_id")["rating"].mean()
        df["rating_normalized"] = df.apply(
            lambda x: x["rating"] - user_means[x["user_id"]], axis=1
        )
        metadata["user_means"] = user_means.to_dict()
        metadata["global_mean"] = float(df["rating"].mean())

    elif method == "z_score":
        # Calculer moyenne et std par utilisateur
        user_stats = df.groupby("user_id")["rating"].agg(["mean", "std"])
        user_stats["std"] = user_stats["std"].fillna(1.0)  # Eviter division par zero

        df["rating_normalized"] = df.apply(
            lambda x: (x["rating"] - user_stats.loc[x["user_id"], "mean"]) /
                      user_stats.loc[x["user_id"], "std"],
            axis=1,
        )
        metadata["user_stats"] = user_stats.to_dict()

    elif method == "min_max":
        # Normalisation globale min-max
        min_rating = df["rating"].min()
        max_rating = df["rating"].max()
        df["rating_normalized"] = (df["rating"] - min_rating) / (max_rating - min_rating)
        metadata["min_rating"] = float(min_rating)
        metadata["max_rating"] = float(max_rating)

    else:
        raise ValueError(f"Methode de normalisation inconnue: {method}")

    logger.info(
        "Notes normalisees",
        method=method,
        original_mean=round(df["rating"].mean(), 3),
        normalized_mean=round(df["rating_normalized"].mean(), 3),
    )

    return df, metadata


def compute_popularity(
    df: pd.DataFrame,
    time_decay: bool = True,
    decay_factor: float = 0.95,
    recency_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Calcule les scores de popularite des items.

    La popularite est calculee en combinant le nombre d'interactions,
    la note moyenne, et optionnellement un facteur de decroissance temporelle.

    Args:
        df: DataFrame des interactions.
        time_decay: Appliquer une decroissance temporelle.
        decay_factor: Facteur de decroissance (0-1).
        recency_weight: Poids de la recence vs popularite brute.

    Returns:
        DataFrame avec colonnes: item_id, popularity_score, interaction_count, avg_rating.

    Example:
        >>> popularity_df = compute_popularity(train_df)
        >>> popularity_df.head()
    """
    # Calculer les statistiques par item
    item_stats = df.groupby("item_id").agg(
        interaction_count=("user_id", "count"),
        avg_rating=("rating", "mean"),
        last_timestamp=("timestamp", "max"),
    ).reset_index()

    # Normaliser le nombre d'interactions
    max_count = item_stats["interaction_count"].max()
    item_stats["count_score"] = item_stats["interaction_count"] / max_count

    # Normaliser la note moyenne
    item_stats["rating_score"] = (item_stats["avg_rating"] - 1) / 4  # Assume 1-5 scale

    if time_decay:
        # Appliquer la decroissance temporelle
        max_ts = df["timestamp"].max()
        time_diffs = (max_ts - item_stats["last_timestamp"]) / (24 * 3600)  # Jours
        item_stats["recency_score"] = decay_factor ** time_diffs

        # Combiner les scores
        item_stats["popularity_score"] = (
            (1 - recency_weight) * item_stats["count_score"] * item_stats["rating_score"] +
            recency_weight * item_stats["recency_score"]
        )
    else:
        # Score sans decroissance temporelle
        item_stats["popularity_score"] = (
            item_stats["count_score"] * item_stats["rating_score"]
        )

    # Normaliser le score final
    item_stats["popularity_score"] = (
        item_stats["popularity_score"] / item_stats["popularity_score"].max()
    )

    result = item_stats[["item_id", "popularity_score", "interaction_count", "avg_rating"]]

    logger.info(
        "Popularite calculee",
        n_items=len(result),
        time_decay=time_decay,
        max_popularity=round(result["popularity_score"].max(), 3),
        min_popularity=round(result["popularity_score"].min(), 3),
    )

    return result


def binarize_ratings(
    df: pd.DataFrame,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Convertit les notes en interactions binaires.

    Cette transformation est necessaire pour les modeles de feedback
    implicite qui ne considerent que les interactions positives/negatives.

    Args:
        df: DataFrame avec colonne 'rating'.
        threshold: Seuil pour considerer une note comme positive.

    Returns:
        DataFrame avec colonne 'interaction' binaire ajoutee.
    """
    df = df.copy()
    df["interaction"] = (df["rating"] >= threshold).astype(np.float32)

    positive_ratio = df["interaction"].mean()

    logger.info(
        "Notes binarisees",
        threshold=threshold,
        positive_ratio=round(positive_ratio, 3),
        n_positive=int(df["interaction"].sum()),
        n_total=len(df),
    )

    return df


def build_confidence_matrix(
    df: pd.DataFrame,
    alpha: float = 40.0,
) -> csr_matrix:
    """
    Construit une matrice de confiance pour le feedback implicite.

    Dans le modele de feedback implicite, la confiance augmente avec
    le nombre d'interactions ou la valeur de la note.

    Args:
        df: DataFrame des interactions.
        alpha: Parametre de mise a l'echelle de la confiance.

    Returns:
        Matrice sparse de confiance.

    Note:
        Confiance = 1 + alpha * interaction_value
    """
    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1

    # Utiliser rating ou interaction comme base
    if "interaction" in df.columns:
        values = 1 + alpha * df["interaction"].values
    else:
        # Normaliser les ratings pour la confiance
        values = 1 + alpha * (df["rating"].values / df["rating"].max())

    confidence_matrix = csr_matrix(
        (values.astype(np.float32), (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    )

    logger.info(
        "Matrice de confiance construite",
        alpha=alpha,
        mean_confidence=round(values.mean(), 2),
        max_confidence=round(values.max(), 2),
    )

    return confidence_matrix


def prepare_features_metadata(
    train_df: pd.DataFrame,
    user_encoder: Any,
    item_encoder: Any,
    feedback_type: str,
) -> Dict[str, Any]:
    """
    Prepare les metadonnees des features pour le serving.

    Ces metadonnees sont utilisees lors de l'inference pour
    decoder les predictions et valider les entrees.

    Args:
        train_df: DataFrame d'entrainement.
        user_encoder: Encodeur des utilisateurs.
        item_encoder: Encodeur des items.
        feedback_type: Type de feedback ('explicit' ou 'implicit').

    Returns:
        Dictionnaire de metadonnees.
    """
    metadata = {
        "n_users": len(user_encoder.classes_),
        "n_items": len(item_encoder.classes_),
        "n_interactions": len(train_df),
        "feedback_type": feedback_type,
        "user_ids": list(map(int, user_encoder.classes_)),
        "item_ids": list(map(int, item_encoder.classes_)),
        "rating_stats": {
            "min": float(train_df["rating"].min()),
            "max": float(train_df["rating"].max()),
            "mean": float(train_df["rating"].mean()),
            "std": float(train_df["rating"].std()),
        },
    }

    logger.info(
        "Metadonnees des features preparees",
        n_users=metadata["n_users"],
        n_items=metadata["n_items"],
    )

    return metadata


def load_params() -> dict:
    """
    Charge les parametres depuis params.yaml.

    Returns:
        Dictionnaire des parametres.
    """
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Point d'entree principal pour la construction des features.

    Cette fonction est appelee par DVC pour construire les features
    selon les parametres definis dans params.yaml.
    """
    # Charger les parametres
    params = load_params()
    data_params = params.get("data", {})
    features_params = params.get("features", {})
    feedback_params = params.get("feedback", {})

    processed_dir = PROJECT_ROOT / data_params.get("processed_dir", "data/processed")

    normalize = features_params.get("normalize_ratings", True)
    compute_pop = features_params.get("compute_popularity", True)
    popularity_decay = features_params.get("popularity_decay", 0.95)
    feedback_type = feedback_params.get("type", "explicit")

    logger.info(
        "Demarrage de la construction des features",
        normalize=normalize,
        compute_popularity=compute_pop,
        feedback_type=feedback_type,
    )

    # Charger les donnees
    train_df = load_parquet(processed_dir / "train.parquet")
    val_df = load_parquet(processed_dir / "val.parquet")
    user_encoder = load_joblib(processed_dir / "user_encoder.joblib")
    item_encoder = load_joblib(processed_dir / "item_encoder.joblib")

    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)

    # Determiner la colonne de valeur
    value_col = "rating"
    if feedback_type == "implicit":
        train_df = binarize_ratings(train_df, feedback_params.get("rating_threshold", 3.5))
        val_df = binarize_ratings(val_df, feedback_params.get("rating_threshold", 3.5))
        value_col = "interaction"

    # Construire les matrices d'interactions
    train_matrix = build_interaction_matrix(
        train_df, n_users, n_items, value_col=value_col
    )
    val_matrix = build_interaction_matrix(
        val_df, n_users, n_items, value_col=value_col
    )

    # Sauvegarder les matrices
    save_sparse_matrix(train_matrix, processed_dir / "train_matrix.npz")
    save_sparse_matrix(val_matrix, processed_dir / "val_matrix.npz")

    # Calculer la popularite si demande
    if compute_pop:
        popularity_df = compute_popularity(
            train_df,
            time_decay=True,
            decay_factor=popularity_decay,
        )
        save_parquet(popularity_df, processed_dir / "item_popularity.parquet")

    # Preparer les metadonnees
    metadata = prepare_features_metadata(
        train_df, user_encoder, item_encoder, feedback_type
    )
    save_json(metadata, processed_dir / "feature_metadata.json")

    logger.info(
        "Construction des features terminee",
        train_matrix_path=str(processed_dir / "train_matrix.npz"),
        val_matrix_path=str(processed_dir / "val_matrix.npz"),
    )


if __name__ == "__main__":
    main()
