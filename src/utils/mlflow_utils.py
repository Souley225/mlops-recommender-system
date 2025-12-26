# -*- coding: utf-8 -*-
"""
Module d'integration MLflow pour le systeme de recommandation.

Ce module fournit des fonctions utilitaires pour l'integration avec MLflow,
incluant la configuration des experiences, le logging des metriques et artefacts,
et la gestion du registre de modeles.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging import get_logger

logger = get_logger(__name__)


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Configure MLflow pour une experience.

    Cette fonction initialise la connexion a MLflow, cree l'experience
    si elle n'existe pas, et retourne l'ID de l'experience.

    Args:
        experiment_name: Nom de l'experience MLflow.
        tracking_uri: URI du serveur MLflow. Si None, utilise la variable
                     d'environnement MLFLOW_TRACKING_URI ou le stockage local.
        tags: Tags optionnels a ajouter a l'experience.

    Returns:
        ID de l'experience MLflow.
    """
    # Configurer l'URI de tracking
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Creer ou recuperer l'experience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags=tags or {},
        )
        logger.info(
            "Experience MLflow creee",
            experiment_name=experiment_name,
            experiment_id=experiment_id,
        )
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            "Experience MLflow existante",
            experiment_name=experiment_name,
            experiment_id=experiment_id,
        )

    mlflow.set_experiment(experiment_name)
    return experiment_id


@contextmanager
def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun, None, None]:
    """
    Demarre un run MLflow dans un contexte.

    Cette fonction fournit un gestionnaire de contexte pour les runs MLflow,
    garantissant la fermeture propre du run meme en cas d'erreur.

    Args:
        run_name: Nom optionnel du run.
        tags: Tags optionnels a ajouter au run.
        nested: Si True, permet les runs imbriques.

    Yields:
        L'objet ActiveRun de MLflow.

    Example:
        >>> with start_run(run_name="entrainement_als") as run:
        ...     mlflow.log_param("embedding_dim", 64)
        ...     mlflow.log_metric("ndcg", 0.85)
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        logger.info(
            "Run MLflow demarre",
            run_id=run.info.run_id,
            run_name=run_name,
        )
        try:
            yield run
        except Exception as e:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            logger.error(
                "Erreur dans le run MLflow",
                run_id=run.info.run_id,
                error=str(e),
            )
            raise
        else:
            mlflow.set_tag("status", "success")
            logger.info(
                "Run MLflow termine avec succes",
                run_id=run.info.run_id,
            )


def log_params(params: Dict[str, Any]) -> None:
    """
    Log plusieurs parametres a la fois.

    Args:
        params: Dictionnaire de parametres a logger.
    """
    # Convertir les valeurs en types compatibles MLflow
    safe_params: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (list, dict)):
            safe_params[key] = str(value)
        else:
            safe_params[key] = value

    mlflow.log_params(safe_params)
    logger.debug("Parametres logges", count=len(params))


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """
    Log plusieurs metriques a la fois.

    Args:
        metrics: Dictionnaire de metriques a logger.
        step: Etape optionnelle pour les metriques temporelles.
    """
    mlflow.log_metrics(metrics, step=step)
    logger.debug("Metriques loggees", count=len(metrics), step=step)


def log_artifact(
    local_path: Union[str, Path],
    artifact_path: Optional[str] = None,
) -> None:
    """
    Log un artefact (fichier ou repertoire).

    Args:
        local_path: Chemin local vers l'artefact.
        artifact_path: Chemin de destination dans MLflow.
    """
    mlflow.log_artifact(str(local_path), artifact_path)
    logger.debug(
        "Artefact logge",
        local_path=str(local_path),
        artifact_path=artifact_path,
    )


def log_model(
    model: Any,
    artifact_path: str,
    registered_model_name: Optional[str] = None,
    **kwargs: Any,
) -> mlflow.models.model.ModelInfo:
    """
    Log un modele avec MLflow.

    Cette fonction sauvegarde le modele comme artefact MLflow et optionnellement
    l'enregistre dans le registre de modeles.

    Args:
        model: Modele a logger (doit etre compatible avec mlflow.pyfunc).
        artifact_path: Chemin de l'artefact dans MLflow.
        registered_model_name: Nom pour l'enregistrement dans le registre.
        **kwargs: Arguments supplementaires pour mlflow.pyfunc.log_model.

    Returns:
        Informations sur le modele logue.
    """
    model_info = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        registered_model_name=registered_model_name,
        **kwargs,
    )
    logger.info(
        "Modele logue",
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    return model_info


def load_model(
    model_uri: str,
) -> Any:
    """
    Charge un modele depuis MLflow.

    Args:
        model_uri: URI du modele (runs:/<run_id>/<path> ou models:/<name>/<stage>).

    Returns:
        Le modele charge.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Modele charge", model_uri=model_uri)
    return model


def get_best_run(
    experiment_name: str,
    metric: str,
    maximize: bool = True,
) -> Optional[mlflow.entities.Run]:
    """
    Recupere le meilleur run d'une experience selon une metrique.

    Args:
        experiment_name: Nom de l'experience.
        metric: Nom de la metrique a optimiser.
        maximize: Si True, cherche le maximum. Sinon, le minimum.

    Returns:
        Le run avec la meilleure valeur de metrique, ou None si aucun run.
    """
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(
            "Experience non trouvee",
            experiment_name=experiment_name,
        )
        return None

    order = "DESC" if maximize else "ASC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        logger.warning(
            "Aucun run trouve",
            experiment_name=experiment_name,
            metric=metric,
        )
        return None

    best_run = runs[0]
    logger.info(
        "Meilleur run trouve",
        run_id=best_run.info.run_id,
        metric=metric,
        value=best_run.data.metrics.get(metric),
    )
    return best_run


def register_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> mlflow.entities.model_registry.ModelVersion:
    """
    Enregistre un modele dans le registre MLflow.

    Args:
        run_id: ID du run contenant le modele.
        artifact_path: Chemin de l'artefact du modele dans le run.
        model_name: Nom du modele dans le registre.
        tags: Tags optionnels pour la version du modele.

    Returns:
        La version du modele enregistree.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri, model_name)

    if tags:
        client = MlflowClient()
        for key, value in tags.items():
            client.set_model_version_tag(
                model_name,
                model_version.version,
                key,
                value,
            )

    logger.info(
        "Modele enregistre",
        model_name=model_name,
        version=model_version.version,
    )
    return model_version


def transition_model_stage(
    model_name: str,
    version: int,
    stage: str,
    archive_existing: bool = True,
) -> mlflow.entities.model_registry.ModelVersion:
    """
    Change le stage d'une version de modele.

    Args:
        model_name: Nom du modele dans le registre.
        version: Numero de version du modele.
        stage: Nouveau stage (Staging, Production, Archived).
        archive_existing: Archiver les versions existantes dans ce stage.

    Returns:
        La version du modele mise a jour.
    """
    client = MlflowClient()
    model_version = client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info(
        "Stage du modele modifie",
        model_name=model_name,
        version=version,
        stage=stage,
    )
    return model_version
