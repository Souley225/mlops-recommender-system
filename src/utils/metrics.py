# -*- coding: utf-8 -*-
"""
Module de metriques d'evaluation pour le systeme de recommandation.

Ce module implemente les metriques standard pour evaluer la qualite
des recommandations: RMSE pour les predictions explicites, et les
metriques de ranking (Precision@K, Recall@K, NDCG@K, MAP@K, MRR).
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Calcule la Root Mean Square Error (RMSE).

    Cette metrique est utilisee pour evaluer la precision des predictions
    de notes dans les systemes de recommandation explicites.

    Args:
        y_true: Notes reelles.
        y_pred: Notes predites.

    Returns:
        Valeur RMSE (plus basse = meilleur).

    Example:
        >>> y_true = np.array([4.0, 3.0, 5.0])
        >>> y_pred = np.array([3.5, 3.2, 4.8])
        >>> rmse(y_true, y_pred)
        0.374...
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Calcule la Mean Absolute Error (MAE).

    Alternative a RMSE moins sensible aux valeurs aberrantes.

    Args:
        y_true: Notes reelles.
        y_pred: Notes predites.

    Returns:
        Valeur MAE (plus basse = meilleur).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(
    relevant_items: Set[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Calcule la Precision@K.

    La precision mesure la proportion d'items recommandes qui sont pertinents
    parmi les K premiers items recommandes.

    Args:
        relevant_items: Ensemble des items pertinents pour l'utilisateur.
        recommended_items: Liste ordonnee des items recommandes.
        k: Nombre d'items a considerer.

    Returns:
        Valeur Precision@K entre 0 et 1.

    Example:
        >>> relevant = {1, 2, 3, 4, 5}
        >>> recommended = [1, 6, 2, 7, 3]
        >>> precision_at_k(relevant, recommended, k=5)
        0.6
    """
    if k <= 0:
        return 0.0

    top_k = recommended_items[:k]
    if not top_k:
        return 0.0

    hits = len(set(top_k) & relevant_items)
    return hits / k


def recall_at_k(
    relevant_items: Set[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Calcule le Recall@K.

    Le recall mesure la proportion d'items pertinents qui ont ete recommandes
    parmi les K premiers items.

    Args:
        relevant_items: Ensemble des items pertinents pour l'utilisateur.
        recommended_items: Liste ordonnee des items recommandes.
        k: Nombre d'items a considerer.

    Returns:
        Valeur Recall@K entre 0 et 1.

    Example:
        >>> relevant = {1, 2, 3, 4, 5}
        >>> recommended = [1, 6, 2, 7, 3]
        >>> recall_at_k(relevant, recommended, k=5)
        0.6
    """
    if not relevant_items or k <= 0:
        return 0.0

    top_k = recommended_items[:k]
    if not top_k:
        return 0.0

    hits = len(set(top_k) & relevant_items)
    return hits / len(relevant_items)


def dcg_at_k(
    relevances: List[float],
    k: int,
) -> float:
    """
    Calcule le Discounted Cumulative Gain@K.

    Le DCG mesure la qualite du ranking en tenant compte de la position
    des items pertinents (les items pertinents en haut du classement
    contribuent plus au score).

    Args:
        relevances: Liste des scores de pertinence dans l'ordre du ranking.
        k: Nombre d'items a considerer.

    Returns:
        Valeur DCG@K.
    """
    if k <= 0 or not relevances:
        return 0.0

    relevances = relevances[:k]
    gains = np.array(relevances)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    relevant_items: Set[int],
    recommended_items: List[int],
    k: int,
    relevance_scores: Optional[Dict[int, float]] = None,
) -> float:
    """
    Calcule le Normalized Discounted Cumulative Gain@K.

    Le NDCG normalise le DCG par le DCG ideal (si les items etaient
    parfaitement ordonnes), donnant un score entre 0 et 1.

    Args:
        relevant_items: Ensemble des items pertinents pour l'utilisateur.
        recommended_items: Liste ordonnee des items recommandes.
        k: Nombre d'items a considerer.
        relevance_scores: Scores de pertinence optionnels par item.
                         Si None, utilise 1.0 pour les items pertinents.

    Returns:
        Valeur NDCG@K entre 0 et 1.

    Example:
        >>> relevant = {1, 2, 3}
        >>> recommended = [1, 4, 2, 5, 3]
        >>> ndcg_at_k(relevant, recommended, k=5)
        0.93...
    """
    if not relevant_items or k <= 0:
        return 0.0

    top_k = recommended_items[:k]
    if not top_k:
        return 0.0

    # Calculer les relevances pour le ranking actuel
    if relevance_scores is None:
        relevances = [1.0 if item in relevant_items else 0.0 for item in top_k]
    else:
        relevances = [relevance_scores.get(item, 0.0) for item in top_k]

    dcg = dcg_at_k(relevances, k)

    # Calculer le DCG ideal
    if relevance_scores is None:
        ideal_relevances = [1.0] * min(len(relevant_items), k)
    else:
        ideal_relevances = sorted(
            [relevance_scores.get(item, 0.0) for item in relevant_items],
            reverse=True,
        )[:k]

    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(
    relevant_items: Set[int],
    recommended_items: List[int],
    k: int,
) -> float:
    """
    Calcule l'Average Precision@K.

    L'AP mesure la precision moyenne aux differentes positions de rappel,
    donnant plus de poids aux items pertinents trouves tot dans le ranking.

    Args:
        relevant_items: Ensemble des items pertinents pour l'utilisateur.
        recommended_items: Liste ordonnee des items recommandes.
        k: Nombre d'items a considerer.

    Returns:
        Valeur AP@K entre 0 et 1.
    """
    if not relevant_items or k <= 0:
        return 0.0

    top_k = recommended_items[:k]
    if not top_k:
        return 0.0

    score = 0.0
    num_hits = 0

    for i, item in enumerate(top_k):
        if item in relevant_items:
            num_hits += 1
            score += num_hits / (i + 1)

    if num_hits == 0:
        return 0.0

    return score / min(len(relevant_items), k)


def mean_average_precision_at_k(
    all_relevant_items: List[Set[int]],
    all_recommended_items: List[List[int]],
    k: int,
) -> float:
    """
    Calcule le Mean Average Precision@K sur plusieurs utilisateurs.

    Args:
        all_relevant_items: Liste des ensembles d'items pertinents par utilisateur.
        all_recommended_items: Liste des listes d'items recommandes par utilisateur.
        k: Nombre d'items a considerer.

    Returns:
        Valeur MAP@K entre 0 et 1.
    """
    if not all_relevant_items:
        return 0.0

    ap_scores = [
        average_precision_at_k(relevant, recommended, k)
        for relevant, recommended in zip(all_relevant_items, all_recommended_items)
    ]
    return float(np.mean(ap_scores))


def reciprocal_rank(
    relevant_items: Set[int],
    recommended_items: List[int],
) -> float:
    """
    Calcule le Reciprocal Rank.

    Le RR est l'inverse du rang du premier item pertinent dans le ranking.

    Args:
        relevant_items: Ensemble des items pertinents pour l'utilisateur.
        recommended_items: Liste ordonnee des items recommandes.

    Returns:
        Valeur RR entre 0 et 1.
    """
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(
    all_relevant_items: List[Set[int]],
    all_recommended_items: List[List[int]],
) -> float:
    """
    Calcule le Mean Reciprocal Rank sur plusieurs utilisateurs.

    Args:
        all_relevant_items: Liste des ensembles d'items pertinents par utilisateur.
        all_recommended_items: Liste des listes d'items recommandes par utilisateur.

    Returns:
        Valeur MRR entre 0 et 1.
    """
    if not all_relevant_items:
        return 0.0

    rr_scores = [
        reciprocal_rank(relevant, recommended)
        for relevant, recommended in zip(all_relevant_items, all_recommended_items)
    ]
    return float(np.mean(rr_scores))


def compute_all_metrics(
    all_relevant_items: List[Set[int]],
    all_recommended_items: List[List[int]],
    k_values: List[int],
    y_true: Optional[NDArray[np.float64]] = None,
    y_pred: Optional[NDArray[np.float64]] = None,
) -> Dict[str, float]:
    """
    Calcule toutes les metriques d'evaluation.

    Cette fonction est le point d'entree principal pour l'evaluation,
    calculant toutes les metriques de ranking et optionnellement RMSE/MAE.

    Args:
        all_relevant_items: Liste des ensembles d'items pertinents par utilisateur.
        all_recommended_items: Liste des listes d'items recommandes par utilisateur.
        k_values: Liste des valeurs de K pour les metriques @K.
        y_true: Notes reelles optionnelles pour RMSE/MAE.
        y_pred: Notes predites optionnelles pour RMSE/MAE.

    Returns:
        Dictionnaire contenant toutes les metriques calculees.

    Example:
        >>> relevant = [{1, 2, 3}, {4, 5}]
        >>> recommended = [[1, 4, 2], [5, 1, 4]]
        >>> metrics = compute_all_metrics(relevant, recommended, k_values=[3, 5])
        >>> metrics["ndcg_at_3"]
        0.87...
    """
    metrics: Dict[str, float] = {}

    # Metriques de prediction (si disponibles)
    if y_true is not None and y_pred is not None:
        metrics["rmse"] = rmse(y_true, y_pred)
        metrics["mae"] = mae(y_true, y_pred)

    # Metriques de ranking pour chaque K
    for k in k_values:
        # Precision@K
        precision_scores = [
            precision_at_k(relevant, recommended, k)
            for relevant, recommended in zip(all_relevant_items, all_recommended_items)
        ]
        metrics[f"precision_at_{k}"] = float(np.mean(precision_scores))

        # Recall@K
        recall_scores = [
            recall_at_k(relevant, recommended, k)
            for relevant, recommended in zip(all_relevant_items, all_recommended_items)
        ]
        metrics[f"recall_at_{k}"] = float(np.mean(recall_scores))

        # NDCG@K
        ndcg_scores = [
            ndcg_at_k(relevant, recommended, k)
            for relevant, recommended in zip(all_relevant_items, all_recommended_items)
        ]
        metrics[f"ndcg_at_{k}"] = float(np.mean(ndcg_scores))

        # MAP@K
        metrics[f"map_at_{k}"] = mean_average_precision_at_k(
            all_relevant_items, all_recommended_items, k
        )

    # MRR (independant de K)
    metrics["mrr"] = mean_reciprocal_rank(all_relevant_items, all_recommended_items)

    return metrics
