# -*- coding: utf-8 -*-
"""
Module d'operations d'entree/sortie pour le systeme de recommandation.

Ce module fournit des fonctions utilitaires pour charger et sauvegarder
differents formats de fichiers: CSV, JSON, Parquet, Pickle, Joblib.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from scipy import sparse


def load_csv(
    filepath: Union[str, Path],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Charge un fichier CSV dans un DataFrame pandas.

    Args:
        filepath: Chemin vers le fichier CSV.
        **kwargs: Arguments supplementaires pour pandas.read_csv.

    Returns:
        DataFrame contenant les donnees du fichier CSV.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier CSV non trouve: {filepath}")
    return pd.read_csv(filepath, **kwargs)


def save_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    **kwargs: Any,
) -> None:
    """
    Sauvegarde un DataFrame pandas dans un fichier CSV.

    Args:
        df: DataFrame a sauvegarder.
        filepath: Chemin de destination.
        index: Inclure l'index dans le fichier. Par defaut: False.
        **kwargs: Arguments supplementaires pour DataFrame.to_csv.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, **kwargs)


def load_parquet(
    filepath: Union[str, Path],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Charge un fichier Parquet dans un DataFrame pandas.

    Le format Parquet est prefere pour les donnees volumineuses car il offre
    une meilleure compression et des performances de lecture superieures.

    Args:
        filepath: Chemin vers le fichier Parquet.
        **kwargs: Arguments supplementaires pour pandas.read_parquet.

    Returns:
        DataFrame contenant les donnees du fichier Parquet.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier Parquet non trouve: {filepath}")
    return pd.read_parquet(filepath, **kwargs)


def save_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    **kwargs: Any,
) -> None:
    """
    Sauvegarde un DataFrame pandas dans un fichier Parquet.

    Args:
        df: DataFrame a sauvegarder.
        filepath: Chemin de destination.
        **kwargs: Arguments supplementaires pour DataFrame.to_parquet.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, **kwargs)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge un fichier JSON dans un dictionnaire Python.

    Args:
        filepath: Chemin vers le fichier JSON.

    Returns:
        Dictionnaire contenant les donnees du fichier JSON.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier JSON non trouve: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Union[Dict[str, Any], List[Any]],
    filepath: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Sauvegarde un dictionnaire ou une liste dans un fichier JSON.

    Args:
        data: Donnees a sauvegarder (dictionnaire ou liste).
        filepath: Chemin de destination.
        indent: Indentation pour le formatage. Par defaut: 2.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_joblib(filepath: Union[str, Path]) -> Any:
    """
    Charge un objet Python serialise avec joblib.

    Joblib est optimise pour les objets numpy et sklearn, offrant
    une meilleure compression et des performances superieures a pickle.

    Args:
        filepath: Chemin vers le fichier joblib.

    Returns:
        L'objet Python deserialise.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier Joblib non trouve: {filepath}")
    # Import model classes before loading to ensure they're registered with Python
    # This is necessary for joblib to find the class when deserializing
    try:
        from src.models.model_classes import PopularityModel, ALSModel  # noqa: F401
    except ImportError:
        pass  # Model classes might not be needed for all joblib files
    return joblib.load(filepath)


def save_joblib(
    obj: Any,
    filepath: Union[str, Path],
    compress: int = 3,
) -> None:
    """
    Sauvegarde un objet Python avec joblib.

    Args:
        obj: Objet Python a sauvegarder.
        filepath: Chemin de destination.
        compress: Niveau de compression (0-9). Par defaut: 3.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath, compress=compress)


def load_sparse_matrix(filepath: Union[str, Path]) -> sparse.csr_matrix:
    """
    Charge une matrice sparse depuis un fichier NPZ.

    Args:
        filepath: Chemin vers le fichier NPZ.

    Returns:
        Matrice sparse au format CSR.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier NPZ non trouve: {filepath}")
    return sparse.load_npz(filepath)


def save_sparse_matrix(
    matrix: sparse.spmatrix,
    filepath: Union[str, Path],
) -> None:
    """
    Sauvegarde une matrice sparse dans un fichier NPZ.

    Args:
        matrix: Matrice sparse a sauvegarder.
        filepath: Chemin de destination.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convertir en CSR si necessaire pour une sauvegarde optimale
    if not isinstance(matrix, sparse.csr_matrix):
        matrix = matrix.tocsr()
    sparse.save_npz(filepath, matrix)
