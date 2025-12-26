# -*- coding: utf-8 -*-
"""
Module de creation du dataset d'interactions.

Ce module charge les donnees brutes MovieLens, les nettoie et les transforme
en un format standardise pour l'entrainement des modeles de recommandation.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml

from src.utils.io import load_csv, save_parquet
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR

logger = get_logger(__name__)


def load_movielens_ratings(
    data_dir: Path,
) -> pd.DataFrame:
    """
    Charge le fichier ratings.csv de MovieLens.

    Cette fonction lit les donnees brutes de notes et effectue
    un nettoyage de base.

    Args:
        data_dir: Repertoire contenant les fichiers MovieLens.

    Returns:
        DataFrame avec les colonnes: userId, movieId, rating, timestamp.

    Raises:
        FileNotFoundError: Si le fichier ratings.csv n'existe pas.
    """
    ratings_path = data_dir / "ratings.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Fichier ratings.csv non trouve dans {data_dir}"
        )

    df = load_csv(ratings_path)

    logger.info(
        "Fichier ratings charge",
        n_rows=len(df),
        n_users=df["userId"].nunique(),
        n_items=df["movieId"].nunique(),
    )

    return df


def load_movielens_movies(
    data_dir: Path,
) -> pd.DataFrame:
    """
    Charge le fichier movies.csv de MovieLens.

    Cette fonction lit les metadonnees des films (titres, genres).

    Args:
        data_dir: Repertoire contenant les fichiers MovieLens.

    Returns:
        DataFrame avec les colonnes: movieId, title, genres.
    """
    movies_path = data_dir / "movies.csv"

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Fichier movies.csv non trouve dans {data_dir}"
        )

    df = load_csv(movies_path)

    logger.info(
        "Fichier movies charge",
        n_movies=len(df),
    )

    return df


def clean_ratings(
    df: pd.DataFrame,
    min_rating: float = 0.5,
    max_rating: float = 5.0,
) -> pd.DataFrame:
    """
    Nettoie le DataFrame des notes.

    Cette fonction supprime les valeurs invalides et les doublons.

    Args:
        df: DataFrame des notes brutes.
        min_rating: Note minimale valide.
        max_rating: Note maximale valide.

    Returns:
        DataFrame nettoye.
    """
    initial_count = len(df)

    # Supprimer les lignes avec valeurs manquantes
    df = df.dropna()

    # Filtrer les notes invalides
    df = df[
        (df["rating"] >= min_rating) &
        (df["rating"] <= max_rating)
    ]

    # Supprimer les doublons (garder la derniere note pour chaque paire user-item)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["userId", "movieId"],
        keep="last",
    )

    final_count = len(df)
    removed = initial_count - final_count

    if removed > 0:
        logger.info(
            "Nettoyage des notes",
            initial_count=initial_count,
            final_count=final_count,
            removed=removed,
        )

    return df


def convert_to_implicit(
    df: pd.DataFrame,
    rating_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Convertit les notes explicites en feedback implicite.

    Les notes superieures au seuil sont considerees comme des interactions
    positives (1), les autres comme negatives (0).

    Args:
        df: DataFrame avec notes explicites.
        rating_threshold: Seuil pour considerer une note comme positive.

    Returns:
        DataFrame avec colonne 'interaction' binaire.
    """
    df = df.copy()
    df["interaction"] = (df["rating"] >= rating_threshold).astype(int)

    positive_count = df["interaction"].sum()
    total_count = len(df)

    logger.info(
        "Conversion en feedback implicite",
        threshold=rating_threshold,
        positive_interactions=positive_count,
        total_interactions=total_count,
        positive_ratio=round(positive_count / total_count, 3),
    )

    return df


def create_interactions_dataset(
    ratings_df: pd.DataFrame,
    feedback_type: str = "explicit",
    rating_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Cree le dataset d'interactions standardise.

    Cette fonction transforme les donnees brutes en un format standardise
    pour l'entrainement, avec des colonnes consistantes.

    Args:
        ratings_df: DataFrame des notes nettoyees.
        feedback_type: Type de feedback ('explicit' ou 'implicit').
        rating_threshold: Seuil pour la conversion en implicite.

    Returns:
        DataFrame avec colonnes: user_id, item_id, rating, timestamp, [interaction].
    """
    # Renommer les colonnes pour un format standardise
    df = ratings_df.rename(columns={
        "userId": "user_id",
        "movieId": "item_id",
    })

    # Convertir en feedback implicite si necessaire
    if feedback_type == "implicit":
        df = convert_to_implicit(df, rating_threshold)

    # Trier par timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Dataset d'interactions cree",
        n_interactions=len(df),
        n_users=df["user_id"].nunique(),
        n_items=df["item_id"].nunique(),
        feedback_type=feedback_type,
    )

    return df


def create_movies_dataset(
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cree le dataset des films standardise.

    Cette fonction transforme les metadonnees des films en un format
    standardise, extrayant les genres en colonnes separees.

    Args:
        movies_df: DataFrame des films brut.

    Returns:
        DataFrame avec colonnes: item_id, title, genres, year.
    """
    df = movies_df.rename(columns={
        "movieId": "item_id",
    })

    # Extraire l'annee du titre si presente (format: "Title (YYYY)")
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Nettoyer le titre
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

    logger.info(
        "Dataset des films cree",
        n_movies=len(df),
        n_with_year=df["year"].notna().sum(),
    )

    return df


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
    Point d'entree principal pour la creation du dataset.

    Cette fonction est appelee par DVC pour creer le dataset
    selon les parametres definis dans params.yaml.
    """
    # Charger les parametres
    params = load_params()
    data_params = params.get("data", {})
    feedback_params = params.get("feedback", {})

    dataset_name = data_params.get("dataset_name", "ml-latest-small")
    raw_dir = PROJECT_ROOT / data_params.get("raw_dir", "data/raw")
    interim_dir = PROJECT_ROOT / data_params.get("interim_dir", "data/interim")

    feedback_type = feedback_params.get("type", "explicit")
    rating_threshold = feedback_params.get("rating_threshold", 3.5)
    min_rating = feedback_params.get("min_rating", 0.5)
    max_rating = feedback_params.get("max_rating", 5.0)

    logger.info(
        "Demarrage de la creation du dataset",
        dataset_name=dataset_name,
        feedback_type=feedback_type,
    )

    # Charger les donnees brutes
    data_dir = raw_dir / dataset_name
    ratings_df = load_movielens_ratings(data_dir)
    movies_df = load_movielens_movies(data_dir)

    # Nettoyer les notes
    ratings_df = clean_ratings(ratings_df, min_rating, max_rating)

    # Creer les datasets
    interactions_df = create_interactions_dataset(
        ratings_df,
        feedback_type=feedback_type,
        rating_threshold=rating_threshold,
    )
    movies_df = create_movies_dataset(movies_df)

    # Sauvegarder les datasets
    save_parquet(interactions_df, interim_dir / "interactions.parquet")
    save_parquet(movies_df, interim_dir / "movies.parquet")

    logger.info(
        "Creation du dataset terminee",
        interactions_path=str(interim_dir / "interactions.parquet"),
        movies_path=str(interim_dir / "movies.parquet"),
    )


if __name__ == "__main__":
    main()
