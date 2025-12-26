# -*- coding: utf-8 -*-
"""
Module d'entrainement des modeles de recommandation.

Ce module implemente plusieurs algorithmes de recommandation:
- Baseline de popularite
- ALS (Alternating Least Squares) pour feedback implicite
- SVD (Singular Value Decomposition) pour feedback explicite

L'entrainement est integre avec MLflow pour le tracking et Optuna
pour l'optimisation des hyperparametres.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import optuna
import yaml
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from scipy.sparse import csr_matrix
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

from src.utils.io import load_json, load_sparse_matrix, save_joblib, save_json
from src.utils.logging import get_logger
from src.utils.mlflow_utils import log_metrics, log_params, setup_mlflow, start_run
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.utils.reproducibility import ensure_reproducibility

logger = get_logger(__name__)


class PopularityModel:
    """
    Modele baseline base sur la popularite des items.

    Ce modele simple recommande les items les plus populaires,
    servant de reference pour evaluer les modeles plus sophistiques.
    """

    def __init__(self) -> None:
        """Initialise le modele de popularite."""
        self.item_scores: Optional[np.ndarray] = None
        self.n_items: int = 0

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs: Any,
    ) -> "PopularityModel":
        """
        Entraine le modele en calculant la popularite des items.

        La popularite est basee sur le nombre d'interactions par item.

        Args:
            interaction_matrix: Matrice sparse user-item.
            **kwargs: Arguments ignores (compatibilite).

        Returns:
            L'instance du modele entraine.
        """
        self.n_items = interaction_matrix.shape[1]

        # Calculer la popularite comme somme des interactions par item
        item_counts = np.array(interaction_matrix.sum(axis=0)).flatten()
        self.item_scores = item_counts / item_counts.max()

        logger.info(
            "Modele de popularite entraine",
            n_items=self.n_items,
        )

        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_items: Optional[set] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genere des recommandations pour un utilisateur.

        Args:
            user_idx: Index de l'utilisateur (ignore, memes recommandations pour tous).
            n: Nombre de recommandations.
            filter_items: Items a exclure des recommandations.

        Returns:
            Tuple (item_indices, scores).
        """
        if self.item_scores is None:
            raise ValueError("Le modele n'a pas ete entraine")

        scores = self.item_scores.copy()

        # Filtrer les items deja vus
        if filter_items:
            for item_idx in filter_items:
                if item_idx < len(scores):
                    scores[item_idx] = -np.inf

        # Trier et retourner les top-n
        top_indices = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_indices]

        return top_indices, top_scores


class ALSModel:
    """
    Modele ALS (Alternating Least Squares) pour feedback implicite.

    Ce modele factorise la matrice d'interactions en facteurs latents
    pour utilisateurs et items, optimise pour le feedback implicite.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Initialise le modele ALS.

        Args:
            factors: Dimension des facteurs latents.
            regularization: Parametre de regularisation L2.
            iterations: Nombre d'iterations d'entrainement.
            alpha: Parametre de confiance pour feedback implicite.
            use_gpu: Utiliser le GPU si disponible.
            random_state: Graine pour la reproductibilite.
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state

        self.model: Optional[AlternatingLeastSquares] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(
        self,
        interaction_matrix: csr_matrix,
        **kwargs: Any,
    ) -> "ALSModel":
        """
        Entraine le modele ALS.

        Args:
            interaction_matrix: Matrice sparse user-item.
            **kwargs: Arguments supplementaires.

        Returns:
            L'instance du modele entraine.
        """
        # Appliquer le facteur de confiance
        confidence_matrix = interaction_matrix.copy()
        confidence_matrix.data = 1 + self.alpha * confidence_matrix.data

        # Creer et entrainer le modele
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            calculate_training_loss=True,
        )

        # Le modele attend item-user (transpose)
        self.model.fit(confidence_matrix.T.tocsr())

        # Stocker les facteurs
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

        logger.info(
            "Modele ALS entraine",
            factors=self.factors,
            iterations=self.iterations,
            user_factors_shape=self.user_factors.shape,
            item_factors_shape=self.item_factors.shape,
        )

        return self

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        filter_items: Optional[set] = None,
        interaction_matrix: Optional[csr_matrix] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genere des recommandations pour un utilisateur.

        Args:
            user_idx: Index de l'utilisateur.
            n: Nombre de recommandations.
            filter_items: Items a exclure.
            interaction_matrix: Matrice pour filtrer les items vus.

        Returns:
            Tuple (item_indices, scores).
        """
        if self.model is None:
            raise ValueError("Le modele n'a pas ete entraine")

        # Calculer les scores
        user_vector = self.user_factors[user_idx]
        scores = self.item_factors.dot(user_vector)

        # Filtrer les items
        if filter_items:
            for item_idx in filter_items:
                if item_idx < len(scores):
                    scores[item_idx] = -np.inf

        if interaction_matrix is not None:
            seen_items = interaction_matrix[user_idx].indices
            scores[seen_items] = -np.inf

        # Top-n
        top_indices = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_indices]

        return top_indices, top_scores

    def similar_items(
        self,
        item_idx: int,
        n: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trouve les items similaires a un item donne.

        Args:
            item_idx: Index de l'item.
            n: Nombre d'items similaires.

        Returns:
            Tuple (item_indices, similarity_scores).
        """
        if self.item_factors is None:
            raise ValueError("Le modele n'a pas ete entraine")

        item_vector = self.item_factors[item_idx]
        similarities = self.item_factors.dot(item_vector)

        # Exclure l'item lui-meme
        similarities[item_idx] = -np.inf

        top_indices = np.argsort(similarities)[::-1][:n]
        top_scores = similarities[top_indices]

        return top_indices, top_scores

    def get_params(self) -> Dict[str, Any]:
        """Retourne les parametres du modele."""
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "use_gpu": self.use_gpu,
        }


def create_optuna_objective(
    train_matrix: csr_matrix,
    val_matrix: csr_matrix,
    model_type: str,
    metric: str,
    k: int = 10,
) -> Callable[[optuna.Trial], float]:
    """
    Cree une fonction objectif pour Optuna.

    Args:
        train_matrix: Matrice d'entrainement.
        val_matrix: Matrice de validation.
        model_type: Type de modele (als, svd).
        metric: Metrique a optimiser.
        k: K pour les metriques @K.

    Returns:
        Fonction objectif pour Optuna.
    """
    from src.models.evaluate import evaluate_model

    def objective(trial: optuna.Trial) -> float:
        # Suggerer les hyperparametres
        factors = trial.suggest_int("factors", 16, 128, step=16)
        regularization = trial.suggest_float("regularization", 0.001, 0.1, log=True)
        iterations = trial.suggest_int("iterations", 5, 30, step=5)

        if model_type == "als":
            alpha = trial.suggest_int("alpha", 10, 100, step=10)
            model = ALSModel(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
            )
        else:
            model = ALSModel(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
            )

        # Entrainer
        model.fit(train_matrix)

        # Evaluer
        metrics = evaluate_model(model, train_matrix, val_matrix, k_values=[k])

        return metrics.get(f"ndcg_at_{k}", 0.0)

    return objective


def run_optuna_optimization(
    train_matrix: csr_matrix,
    val_matrix: csr_matrix,
    model_type: str,
    n_trials: int = 20,
    metric: str = "ndcg_at_10",
) -> Dict[str, Any]:
    """
    Execute l'optimisation des hyperparametres avec Optuna.

    Args:
        train_matrix: Matrice d'entrainement.
        val_matrix: Matrice de validation.
        model_type: Type de modele.
        n_trials: Nombre d'essais.
        metric: Metrique a optimiser.

    Returns:
        Meilleurs hyperparametres.
    """
    # Extraire K de la metrique
    k = int(metric.split("_")[-1]) if "_at_" in metric else 10

    objective = create_optuna_objective(
        train_matrix, val_matrix, model_type, metric, k
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Optimisation Optuna terminee",
        best_params=study.best_params,
        best_value=study.best_value,
        n_trials=n_trials,
    )

    return study.best_params


def train_model(
    model_type: str,
    train_matrix: csr_matrix,
    params: Dict[str, Any],
) -> Union[PopularityModel, ALSModel]:
    """
    Entraine un modele avec les parametres specifies.

    Args:
        model_type: Type de modele (popularity, als).
        train_matrix: Matrice d'entrainement.
        params: Parametres du modele.

    Returns:
        Modele entraine.
    """
    if model_type == "popularity":
        model = PopularityModel()
    elif model_type == "als":
        model = ALSModel(
            factors=params.get("embedding_dim", params.get("factors", 64)),
            regularization=params.get("regularization", 0.01),
            iterations=params.get("iterations", 15),
            alpha=params.get("alpha", 40),
            use_gpu=params.get("use_gpu", False),
            random_state=params.get("random_seed", 42),
        )
    else:
        raise ValueError(f"Type de modele inconnu: {model_type}")

    model.fit(train_matrix)
    return model


def load_params() -> dict:
    """Charge les parametres depuis params.yaml."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Point d'entree principal pour l'entrainement.

    Cette fonction est appelee par DVC pour entrainer le modele
    selon les parametres definis dans params.yaml.
    """
    # Charger les parametres
    params = load_params()
    model_params = params.get("model", {})
    train_params = params.get("train", {})
    optuna_params = params.get("optuna", {})

    model_type = model_params.get("type", "als")
    random_seed = train_params.get("random_seed", 42)
    log_to_mlflow = train_params.get("log_to_mlflow", True)
    experiment_name = train_params.get("experiment_name", "mlops-recommender")

    # Configuration de la reproductibilite
    ensure_reproducibility(random_seed)

    logger.info(
        "Demarrage de l'entrainement",
        model_type=model_type,
        random_seed=random_seed,
    )

    # Charger les donnees
    train_matrix = load_sparse_matrix(PROCESSED_DATA_DIR / "train_matrix.npz")
    val_matrix = load_sparse_matrix(PROCESSED_DATA_DIR / "val_matrix.npz")
    metadata = load_json(PROCESSED_DATA_DIR / "feature_metadata.json")

    # Configuration MLflow
    if log_to_mlflow:
        setup_mlflow(experiment_name)

    # Optimisation Optuna si activee
    if optuna_params.get("enabled", False) and model_type != "popularity":
        logger.info("Demarrage de l'optimisation Optuna")
        best_params = run_optuna_optimization(
            train_matrix,
            val_matrix,
            model_type,
            n_trials=optuna_params.get("n_trials", 20),
            metric=optuna_params.get("metric", "ndcg_at_10"),
        )
        # Fusionner avec les parametres du modele
        model_params.update(best_params)

    # Entrainer le modele final
    with start_run(run_name=f"train_{model_type}") as run:
        if log_to_mlflow:
            log_params({
                "model_type": model_type,
                "n_users": metadata["n_users"],
                "n_items": metadata["n_items"],
                "n_interactions": metadata["n_interactions"],
                **model_params,
            })

        model = train_model(model_type, train_matrix, model_params)

        # Sauvegarder le modele
        save_joblib(model, MODELS_DIR / "model.joblib")

        # Sauvegarder les metriques d'entrainement
        train_metrics = {
            "model_type": model_type,
            "n_users": metadata["n_users"],
            "n_items": metadata["n_items"],
        }
        if hasattr(model, "get_params"):
            train_metrics.update(model.get_params())

        save_json(train_metrics, MODELS_DIR / "train_metrics.json")

        if log_to_mlflow:
            log_metrics({"training_complete": 1.0})

    logger.info(
        "Entrainement termine",
        model_path=str(MODELS_DIR / "model.joblib"),
    )


if __name__ == "__main__":
    main()
