# -*- coding: utf-8 -*-
"""
Module de telechargement du jeu de donnees MovieLens.

Ce module gere le telechargement et l'extraction du jeu de donnees MovieLens
depuis les serveurs GroupLens. Il supporte plusieurs versions du dataset.
"""

import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import yaml

from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT, RAW_DATA_DIR, ensure_dir

logger = get_logger(__name__)

# URLs des jeux de donnees MovieLens disponibles
MOVIELENS_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
}


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 8192,
) -> Path:
    """
    Telecharge un fichier depuis une URL.

    Cette fonction telecharge un fichier en affichant la progression
    et gere les erreurs de connexion.

    Args:
        url: URL du fichier a telecharger.
        destination: Chemin de destination pour le fichier.
        chunk_size: Taille des chunks pour le telechargement.

    Returns:
        Chemin du fichier telecharge.

    Raises:
        Exception: Si le telechargement echoue.
    """
    logger.info(
        "Demarrage du telechargement",
        url=url,
        destination=str(destination),
    )

    ensure_dir(destination.parent)

    try:
        urlretrieve(url, destination)
        logger.info(
            "Telechargement termine",
            destination=str(destination),
            size_mb=round(destination.stat().st_size / (1024 * 1024), 2),
        )
        return destination
    except Exception as e:
        logger.error(
            "Erreur lors du telechargement",
            url=url,
            error=str(e),
        )
        raise


def extract_zip(
    zip_path: Path,
    extract_dir: Path,
) -> Path:
    """
    Extrait une archive ZIP.

    Cette fonction extrait le contenu d'une archive ZIP dans le repertoire
    specifie, supprimant l'archive apres extraction.

    Args:
        zip_path: Chemin de l'archive ZIP.
        extract_dir: Repertoire de destination pour l'extraction.

    Returns:
        Chemin du repertoire extrait.
    """
    logger.info(
        "Extraction de l'archive",
        zip_path=str(zip_path),
        extract_dir=str(extract_dir),
    )

    ensure_dir(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Supprimer l'archive apres extraction
    zip_path.unlink()

    logger.info("Extraction terminee", extract_dir=str(extract_dir))
    return extract_dir


def download_movielens(
    dataset_name: str = "ml-latest-small",
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Telecharge et extrait un jeu de donnees MovieLens.

    Cette fonction est le point d'entree principal pour obtenir les donnees
    MovieLens. Elle verifie si les donnees existent deja et les telecharge
    si necessaire.

    Args:
        dataset_name: Nom du jeu de donnees (ml-latest-small, ml-25m, etc.).
        output_dir: Repertoire de sortie. Par defaut: data/raw.
        force: Si True, retelecharge meme si les donnees existent.

    Returns:
        Chemin du repertoire contenant les donnees extraites.

    Raises:
        ValueError: Si le nom du jeu de donnees n'est pas reconnu.

    Example:
        >>> data_dir = download_movielens("ml-latest-small")
        >>> list(data_dir.glob("*.csv"))
        [PosixPath('data/raw/ml-latest-small/ratings.csv'), ...]
    """
    if dataset_name not in MOVIELENS_URLS:
        raise ValueError(
            f"Jeu de donnees inconnu: {dataset_name}. "
            f"Options disponibles: {list(MOVIELENS_URLS.keys())}"
        )

    if output_dir is None:
        output_dir = RAW_DATA_DIR

    output_dir = Path(output_dir)
    dataset_dir = output_dir / dataset_name

    # Verifier si les donnees existent deja
    if dataset_dir.exists() and not force:
        if (dataset_dir / "ratings.csv").exists():
            logger.info(
                "Donnees deja presentes, telechargement ignore",
                dataset_dir=str(dataset_dir),
            )
            return dataset_dir

    # Telecharger l'archive
    url = MOVIELENS_URLS[dataset_name]
    zip_path = output_dir / f"{dataset_name}.zip"

    # Nettoyer le repertoire existant si force=True
    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)
        logger.info("Repertoire existant supprime", dataset_dir=str(dataset_dir))

    # Telecharger et extraire
    download_file(url, zip_path)
    extract_zip(zip_path, output_dir)

    # Verifier que l'extraction a reussi
    if not (dataset_dir / "ratings.csv").exists():
        raise RuntimeError(
            f"Extraction echouee: fichier ratings.csv non trouve dans {dataset_dir}"
        )

    logger.info(
        "Jeu de donnees MovieLens pret",
        dataset_name=dataset_name,
        dataset_dir=str(dataset_dir),
    )

    return dataset_dir


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
    Point d'entree principal pour le telechargement des donnees.

    Cette fonction est appelee par DVC pour telecharger les donnees
    selon les parametres definis dans params.yaml.
    """
    # Charger les parametres
    params = load_params()
    data_params = params.get("data", {})

    dataset_name = data_params.get("dataset_name", "ml-latest-small")
    raw_dir = PROJECT_ROOT / data_params.get("raw_dir", "data/raw")

    logger.info(
        "Demarrage du pipeline de telechargement",
        dataset_name=dataset_name,
        raw_dir=str(raw_dir),
    )

    # Telecharger les donnees
    dataset_dir = download_movielens(
        dataset_name=dataset_name,
        output_dir=raw_dir,
        force=False,
    )

    logger.info(
        "Pipeline de telechargement termine",
        dataset_dir=str(dataset_dir),
    )


if __name__ == "__main__":
    main()
