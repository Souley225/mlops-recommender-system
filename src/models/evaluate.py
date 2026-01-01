# -*- coding: utf-8 -*-
"""
Module d'evaluation des modeles de recommandation.

Ce module implemente l'evaluation des modeles avec les metriques
standard de ranking: Precision@K, Recall@K, NDCG@K, MAP@K, MRR.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix

from src.utils.io import (
    load_joblib,
    load_parquet,
    load_sparse_matrix,
    save_csv,
    save_json,
)
from src.utils.logging import get_logger
from src.utils.metrics import (
    compute_all_metrics,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_average_precision_at_k,
    mean_reciprocal_rank,
    rmse,
)
from src.utils.mlflow_utils import log_metrics, setup_mlflow, start_run
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
# Import model classes so joblib can find them when deserializing
from src.models.train import PopularityModel, ALSModel  # noqa: F401

logger = get_logger(__name__)


def get_user_test_items(
    test_df: pd.DataFrame,
    user_encoder: Any,
    relevance_threshold: float = 3.5,
) -> Dict[int, Set[int]]:
    """
    Extrait les items pertinents par utilisateur depuis le test set.

    Args:
        test_df: DataFrame de test avec colonnes user_idx, item_idx, rating.
        user_encoder: Encodeur des utilisateurs.
        relevance_threshold: Seuil de note pour considerer un item pertinent.

    Returns:
        Dictionnaire {user_idx: set(item_idx pertinents)}.
    """
    # Filtrer les items pertinents
    relevant_df = test_df[test_df["rating"] >= relevance_threshold]

    # Grouper par utilisateur
    user_items: Dict[int, Set[int]] = {}
    for user_idx, group in relevant_df.groupby("user_idx"):
        user_items[int(user_idx)] = set(group["item_idx"].values)

    logger.info(
        "Items de test extraits",
        n_users=len(user_items),
        avg_items_per_user=round(
            sum(len(items) for items in user_items.values()) / max(len(user_items), 1), 2
        ),
    )

    return user_items


def evaluate_user(
    model: Any,
    user_idx: int,
    relevant_items: Set[int],
    train_matrix: csr_matrix,
    k_values: List[int],
    n_candidates: int = 100,
) -> Dict[str, float]:
    """
    Evalue les recommandations pour un utilisateur.

    Args:
        model: Modele de recommandation.
        user_idx: Index de l'utilisateur.
        relevant_items: Items pertinents pour cet utilisateur.
        train_matrix: Matrice d'entrainement pour filtrer les items vus.
        k_values: Valeurs de K pour les metriques.
        n_candidates: Nombre de candidats a generer.

    Returns:
        Dictionnaire des metriques pour cet utilisateur.
    """
    # Generer les recommandations
    try:
        recommended_items, scores = model.recommend(
            user_idx,
            n=n_candidates,
            interaction_matrix=train_matrix,
        )
    except Exception:
        # Fallback si le modele ne supporte pas interaction_matrix
        seen_items = set(train_matrix[user_idx].indices)
        recommended_items, scores = model.recommend(
            user_idx,
            n=n_candidates,
            filter_items=seen_items,
        )

    recommended_list = list(recommended_items)

    # Calculer les metriques
    user_metrics: Dict[str, float] = {}

    for k in k_values:
        user_metrics[f"precision_at_{k}"] = precision_at_k(relevant_items, recommended_list, k)
        user_metrics[f"recall_at_{k}"] = recall_at_k(relevant_items, recommended_list, k)
        user_metrics[f"ndcg_at_{k}"] = ndcg_at_k(relevant_items, recommended_list, k)

    return user_metrics


def evaluate_model(
    model: Any,
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    k_values: List[int] = [5, 10, 20],
    n_users: Optional[int] = None,
    relevance_threshold: float = 3.5,
) -> Dict[str, float]:
    """
    Evalue un modele sur l'ensemble de test.

    Cette fonction genere des recommandations pour chaque utilisateur
    et calcule les metriques agregees.

    Args:
        model: Modele de recommandation entraine.
        train_matrix: Matrice d'entrainement (pour filtrer les items vus).
        test_matrix: Matrice de test.
        k_values: Valeurs de K pour les metriques @K.
        n_users: Nombre d'utilisateurs a evaluer (None = tous).
        relevance_threshold: Seuil pour considerer une note comme pertinente.

    Returns:
        Dictionnaire des metriques agregees.
    """
    n_total_users = train_matrix.shape[0]

    # Trouver les utilisateurs avec des interactions de test
    test_users = set(test_matrix.nonzero()[0])

    if n_users is not None:
        test_users = set(list(test_users)[:n_users])

    logger.info(
        "Demarrage de l'evaluation",
        n_test_users=len(test_users),
        k_values=k_values,
    )

    all_relevant: List[Set[int]] = []
    all_recommended: List[List[int]] = []

    for user_idx in test_users:
        # Items pertinents dans le test
        test_items = test_matrix[user_idx].indices
        test_ratings = test_matrix[user_idx].data

        relevant_items = set(
            item_idx for item_idx, rating in zip(test_items, test_ratings)
            if rating >= relevance_threshold
        )

        if not relevant_items:
            continue

        # Generer les recommandations
        try:
            recommended_items, _ = model.recommend(
                user_idx,
                n=max(k_values),
                interaction_matrix=train_matrix,
            )
        except TypeError:
            seen_items = set(train_matrix[user_idx].indices)
            recommended_items, _ = model.recommend(
                user_idx,
                n=max(k_values),
                filter_items=seen_items,
            )

        all_relevant.append(relevant_items)
        all_recommended.append(list(recommended_items))

    # Calculer les metriques agregees
    metrics = compute_all_metrics(
        all_relevant,
        all_recommended,
        k_values,
    )

    logger.info(
        "Evaluation terminee",
        n_users_evaluated=len(all_relevant),
        metrics=metrics,
    )

    return metrics


def load_params() -> dict:
    """Charge les parametres depuis params.yaml."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Point d'entree principal pour l'evaluation.

    Cette fonction est appelee par DVC pour evaluer le modele
    selon les parametres definis dans params.yaml.
    """
    # Charger les parametres
    params = load_params()
    eval_params = params.get("eval", {})
    feedback_params = params.get("feedback", {})
    train_params = params.get("train", {})

    k_values = eval_params.get("k_values", [5, 10, 20])
    compute_ranking = eval_params.get("compute_ranking_metrics", True)
    feedback_type = feedback_params.get("type", "explicit")
    relevance_threshold = feedback_params.get("rating_threshold", 3.5)
    log_to_mlflow = train_params.get("log_to_mlflow", True)
    experiment_name = train_params.get("experiment_name", "mlops-recommender")

    logger.info(
        "Demarrage de l'evaluation du modele",
        k_values=k_values,
        feedback_type=feedback_type,
    )

    # Charger le modele et les donnees
    model = load_joblib(MODELS_DIR / "model.joblib")
    train_matrix = load_sparse_matrix(PROCESSED_DATA_DIR / "train_matrix.npz")
    val_matrix = load_sparse_matrix(PROCESSED_DATA_DIR / "val_matrix.npz")
    test_df = load_parquet(PROCESSED_DATA_DIR / "test.parquet")

    # Construire la matrice de test
    n_users, n_items = train_matrix.shape

    if "user_idx" in test_df.columns and "item_idx" in test_df.columns:
        # Filtrer les lignes valides
        valid_test = test_df[
            (test_df["user_idx"] < n_users) &
            (test_df["item_idx"] < n_items)
        ]

        test_matrix = csr_matrix(
            (valid_test["rating"].values,
             (valid_test["user_idx"].values, valid_test["item_idx"].values)),
            shape=(n_users, n_items),
        )
    else:
        # Utiliser val_matrix comme fallback
        test_matrix = val_matrix

    # Evaluer le modele
    metrics = evaluate_model(
        model,
        train_matrix,
        test_matrix,
        k_values=k_values,
        relevance_threshold=relevance_threshold if feedback_type == "explicit" else 0.5,
    )

    # Sauvegarder les metriques
    save_json(metrics, MODELS_DIR / "eval_metrics.json")

    # Sauvegarder les metriques par K pour les plots
    metrics_by_k = []
    for k in k_values:
        metrics_by_k.append({
            "k": k,
            "precision": metrics.get(f"precision_at_{k}", 0),
            "recall": metrics.get(f"recall_at_{k}", 0),
            "ndcg": metrics.get(f"ndcg_at_{k}", 0),
            "map": metrics.get(f"map_at_{k}", 0),
        })
    save_csv(pd.DataFrame(metrics_by_k), MODELS_DIR / "metrics_by_k.csv")

    # Logger dans MLflow
    if log_to_mlflow:
        setup_mlflow(experiment_name)
        with start_run(run_name="evaluate") as run:
            log_metrics(metrics)

    logger.info(
        "Evaluation terminee",
        metrics_path=str(MODELS_DIR / "eval_metrics.json"),
    )


if __name__ == "__main__":
    main()
