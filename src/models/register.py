# -*- coding: utf-8 -*-
"""
Module d'enregistrement des modeles dans MLflow.

Ce module gere l'enregistrement du meilleur modele dans le registre
MLflow, permettant le versioning et le deploiement.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.io import load_joblib, load_json, save_json
from src.utils.logging import get_logger
from src.utils.mlflow_utils import (
    get_best_run,
    register_model,
    setup_mlflow,
    start_run,
    transition_model_stage,
    log_artifact,
    log_params,
)
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

logger = get_logger(__name__)


def package_model_artifacts(
    model_path: Path,
    encoder_paths: Dict[str, Path],
    metadata_path: Path,
    output_dir: Path,
) -> Path:
    """
    Prepare les artefacts du modele pour l'enregistrement.

    Cette fonction copie tous les fichiers necessaires au serving
    dans un repertoire unique pour l'enregistrement MLflow.

    Args:
        model_path: Chemin vers le modele.
        encoder_paths: Chemins vers les encodeurs.
        metadata_path: Chemin vers les metadonnees.
        output_dir: Repertoire de sortie.

    Returns:
        Chemin du repertoire contenant les artefacts.
    """
    import shutil

    artifacts_dir = output_dir / "model_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Copier le modele
    shutil.copy(model_path, artifacts_dir / "model.joblib")

    # Copier les encodeurs
    for name, path in encoder_paths.items():
        if path.exists():
            shutil.copy(path, artifacts_dir / f"{name}.joblib")

    # Copier les metadonnees
    if metadata_path.exists():
        shutil.copy(metadata_path, artifacts_dir / "metadata.json")

    # Copier la popularite si disponible
    popularity_path = PROCESSED_DATA_DIR / "item_popularity.parquet"
    if popularity_path.exists():
        shutil.copy(popularity_path, artifacts_dir / "item_popularity.parquet")

    logger.info(
        "Artefacts du modele prepares",
        artifacts_dir=str(artifacts_dir),
        files=list(f.name for f in artifacts_dir.iterdir()),
    )

    return artifacts_dir


def register_best_model(
    experiment_name: str,
    model_name: str = "als_recommender",
    metric: str = "ndcg_at_10",
    stage: str = "Production",
) -> Optional[Dict[str, Any]]:
    """
    Enregistre le meilleur modele dans le registre MLflow.

    Cette fonction trouve le meilleur run selon une metrique,
    enregistre le modele et le promeut au stage specifie.

    Args:
        experiment_name: Nom de l'experience MLflow.
        model_name: Nom du modele dans le registre.
        metric: Metrique pour determiner le meilleur modele.
        stage: Stage de deploiement (Staging, Production).

    Returns:
        Informations sur le modele enregistre, ou None si echec.
    """
    # Trouver le meilleur run
    best_run = get_best_run(experiment_name, metric, maximize=True)

    if best_run is None:
        logger.warning(
            "Aucun run trouve pour l'enregistrement",
            experiment_name=experiment_name,
        )
        return None

    run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(metric, 0)

    logger.info(
        "Meilleur run identifie",
        run_id=run_id,
        metric=metric,
        value=best_metric,
    )

    # Enregistrer le modele
    try:
        model_version = register_model(
            run_id=run_id,
            artifact_path="model",
            model_name=model_name,
            tags={
                "metric": metric,
                "metric_value": str(best_metric),
            },
        )

        # Promouvoir au stage
        transition_model_stage(
            model_name=model_name,
            version=model_version.version,
            stage=stage,
        )

        result = {
            "model_name": model_name,
            "version": model_version.version,
            "stage": stage,
            "run_id": run_id,
            "metric": metric,
            "metric_value": best_metric,
        }

        logger.info(
            "Modele enregistre avec succes",
            **result,
        )

        return result

    except Exception as e:
        logger.error(
            "Erreur lors de l'enregistrement du modele",
            error=str(e),
        )
        return None


def create_local_registration(
    model_path: Path,
    eval_metrics_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Cree un enregistrement local du modele.

    Utilise comme fallback quand MLflow n'est pas disponible,
    ou pour un deploiement simplifie.

    Args:
        model_path: Chemin vers le modele.
        eval_metrics_path: Chemin vers les metriques d'evaluation.
        output_path: Chemin pour sauvegarder les informations.

    Returns:
        Informations sur le modele enregistre.
    """
    import hashlib
    from datetime import datetime

    # Calculer un hash du modele comme version
    with open(model_path, "rb") as f:
        model_hash = hashlib.md5(f.read()).hexdigest()[:8]

    # Charger les metriques
    eval_metrics = {}
    if eval_metrics_path.exists():
        eval_metrics = load_json(eval_metrics_path)

    registration_info = {
        "model_name": "als_recommender",
        "version": model_hash,
        "registered_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "metrics": eval_metrics,
        "stage": "Production",
    }

    save_json(registration_info, output_path)

    logger.info(
        "Enregistrement local cree",
        output_path=str(output_path),
        version=model_hash,
    )

    return registration_info


def load_params() -> dict:
    """Charge les parametres depuis params.yaml."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Point d'entree principal pour l'enregistrement du modele.

    Cette fonction est appelee par DVC pour enregistrer le modele
    dans le registre MLflow ou localement.
    """
    # Charger les parametres
    params = load_params()
    train_params = params.get("train", {})

    experiment_name = train_params.get("experiment_name", "mlops-recommender")
    log_to_mlflow = train_params.get("log_to_mlflow", True)

    logger.info(
        "Demarrage de l'enregistrement du modele",
        experiment_name=experiment_name,
    )

    # Chemins des artefacts
    model_path = MODELS_DIR / "model.joblib"
    eval_metrics_path = MODELS_DIR / "eval_metrics.json"
    output_path = MODELS_DIR / "registered_model_info.json"

    if not model_path.exists():
        logger.error("Modele non trouve", model_path=str(model_path))
        return

    # Tenter l'enregistrement MLflow
    if log_to_mlflow:
        try:
            setup_mlflow(experiment_name)

            # Logger les artefacts dans un nouveau run
            with start_run(run_name="register") as run:
                log_artifact(model_path)
                log_artifact(eval_metrics_path)

                # Enregistrer les encodeurs
                user_encoder_path = PROCESSED_DATA_DIR / "user_encoder.joblib"
                item_encoder_path = PROCESSED_DATA_DIR / "item_encoder.joblib"

                if user_encoder_path.exists():
                    log_artifact(user_encoder_path)
                if item_encoder_path.exists():
                    log_artifact(item_encoder_path)

            # Tenter l'enregistrement dans le registre
            result = register_best_model(
                experiment_name=experiment_name,
                model_name="als_recommender",
                metric="ndcg_at_10",
                stage="Production",
            )

            if result:
                save_json(result, output_path)
                logger.info("Enregistrement MLflow termine")
                return

        except Exception as e:
            logger.warning(
                "Enregistrement MLflow echoue, utilisation du fallback local",
                error=str(e),
            )

    # Fallback: enregistrement local
    result = create_local_registration(
        model_path=model_path,
        eval_metrics_path=eval_metrics_path,
        output_path=output_path,
    )

    logger.info(
        "Enregistrement termine",
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()
