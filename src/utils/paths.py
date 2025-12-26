# -*- coding: utf-8 -*-
"""
Module de gestion des chemins pour le systeme de recommandation.

Ce module centralise la definition de tous les chemins utilises dans le projet,
garantissant une gestion coherente des repertoires de donnees, modeles et artefacts.
"""

from pathlib import Path
from typing import Optional

# Repertoire racine du projet (deux niveaux au-dessus de ce fichier)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Repertoires de donnees
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Repertoires de modeles et artefacts
MODELS_DIR: Path = PROJECT_ROOT / "models"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Repertoire de logs
LOGS_DIR: Path = PROJECT_ROOT / "logs"


def ensure_dir(path: Path) -> Path:
    """
    Cree un repertoire s'il n'existe pas.

    Cette fonction garantit que le repertoire specifie existe,
    le creant recursivement si necessaire.

    Args:
        path: Chemin du repertoire a creer.

    Returns:
        Le chemin du repertoire cree ou existant.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(
    filename: str,
    subdir: str = "processed",
) -> Path:
    """
    Construit le chemin complet vers un fichier de donnees.

    Cette fonction simplifie l'acces aux fichiers de donnees en construisant
    automatiquement le chemin complet a partir du nom de fichier.

    Args:
        filename: Nom du fichier de donnees.
        subdir: Sous-repertoire (raw, interim, ou processed). Par defaut: processed.

    Returns:
        Chemin complet vers le fichier de donnees.

    Raises:
        ValueError: Si le sous-repertoire n'est pas valide.
    """
    subdirs = {
        "raw": RAW_DATA_DIR,
        "interim": INTERIM_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
    }
    if subdir not in subdirs:
        raise ValueError(
            f"Sous-repertoire invalide: {subdir}. "
            f"Valeurs acceptees: {list(subdirs.keys())}"
        )
    return subdirs[subdir] / filename


def get_model_path(filename: str) -> Path:
    """
    Construit le chemin complet vers un fichier de modele.

    Args:
        filename: Nom du fichier de modele.

    Returns:
        Chemin complet vers le fichier de modele.
    """
    return MODELS_DIR / filename


def get_config_path(filename: str, subdir: Optional[str] = None) -> Path:
    """
    Construit le chemin complet vers un fichier de configuration.

    Args:
        filename: Nom du fichier de configuration.
        subdir: Sous-repertoire optionnel (ex: hydra).

    Returns:
        Chemin complet vers le fichier de configuration.
    """
    if subdir:
        return CONFIGS_DIR / subdir / filename
    return CONFIGS_DIR / filename


def setup_directories() -> None:
    """
    Cree tous les repertoires necessaires au projet.

    Cette fonction initialise la structure de repertoires du projet,
    creant les dossiers de donnees, modeles et logs s'ils n'existent pas.
    """
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        ensure_dir(directory)


# Creer les repertoires au chargement du module
setup_directories()
