# -*- coding: utf-8 -*-
"""
Module de division du dataset en train/validation/test.

Ce module implemente des strategies de division respectant la temporalite
des interactions, essentielle pour une evaluation realiste des systemes
de recommandation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from src.utils.io import load_parquet, save_joblib, save_parquet
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


def filter_cold_start(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """
    Filtre les utilisateurs et items avec trop peu d'interactions.

    Cette strategie reduit le probleme du cold-start en ne gardant que
    les entites avec suffisamment d'historique pour un apprentissage fiable.

    Args:
        df: DataFrame des interactions.
        min_user_interactions: Nombre minimum d'interactions par utilisateur.
        min_item_interactions: Nombre minimum d'interactions par item.

    Returns:
        DataFrame filtre.
    """
    initial_count = len(df)
    initial_users = df["user_id"].nunique()
    initial_items = df["item_id"].nunique()

    # Filtrer iterativement jusqu'a stabilisation
    prev_count = 0
    while len(df) != prev_count:
        prev_count = len(df)

        # Calculer les comptes
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()

        # Filtrer les utilisateurs
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        # Filtrer les items
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)]

    final_count = len(df)
    final_users = df["user_id"].nunique()
    final_items = df["item_id"].nunique()

    logger.info(
        "Filtrage cold-start termine",
        interactions_removed=initial_count - final_count,
        users_removed=initial_users - final_users,
        items_removed=initial_items - final_items,
        final_interactions=final_count,
        final_users=final_users,
        final_items=final_items,
    )

    return df


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise le dataset en respectant l'ordre temporel.

    Cette strategie simule un scenario realiste ou le modele est entraine
    sur des donnees passees et evalue sur des donnees futures.

    Args:
        df: DataFrame des interactions trie par timestamp.
        train_ratio: Proportion pour l'entrainement.
        val_ratio: Proportion pour la validation.
        test_ratio: Proportion pour le test.

    Returns:
        Tuple de DataFrames (train, validation, test).

    Raises:
        ValueError: Si les ratios ne somment pas a 1.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Les ratios doivent sommer a 1.0, obtenu: "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    # S'assurer que le DataFrame est trie par timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Split temporel effectue",
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        train_end_date=pd.to_datetime(train_df["timestamp"].max(), unit="s"),
        val_end_date=pd.to_datetime(val_df["timestamp"].max(), unit="s"),
        test_end_date=pd.to_datetime(test_df["timestamp"].max(), unit="s"),
    )

    return train_df, val_df, test_df


def random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise le dataset de maniere aleatoire.

    Cette strategie est moins realiste mais peut etre utile pour
    certaines evaluations ou comparaisons.

    Args:
        df: DataFrame des interactions.
        train_ratio: Proportion pour l'entrainement.
        val_ratio: Proportion pour la validation.
        test_ratio: Proportion pour le test.
        seed: Graine pour la reproductibilite.

    Returns:
        Tuple de DataFrames (train, validation, test).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Les ratios doivent sommer a 1.0, obtenu: "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    set_seed(seed)

    # Melanger le DataFrame
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Split aleatoire effectue",
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
    )

    return train_df, val_df, test_df


def create_encoders(
    train_df: pd.DataFrame,
) -> Tuple[LabelEncoder, LabelEncoder]:
    """
    Cree des encodeurs pour les IDs utilisateurs et items.

    Ces encodeurs convertissent les IDs originaux en indices
    continus (0, 1, 2, ...) necessaires pour les matrices.

    Args:
        train_df: DataFrame d'entrainement.

    Returns:
        Tuple (user_encoder, item_encoder).
    """
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(train_df["user_id"])
    item_encoder.fit(train_df["item_id"])

    logger.info(
        "Encodeurs crees",
        n_users=len(user_encoder.classes_),
        n_items=len(item_encoder.classes_),
    )

    return user_encoder, item_encoder


def encode_ids(
    df: pd.DataFrame,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    handle_unknown: str = "ignore",
) -> pd.DataFrame:
    """
    Encode les IDs utilisateurs et items.

    Cette fonction convertit les IDs originaux en indices encodes,
    gerant les IDs inconnus selon la strategie specifiee.

    Args:
        df: DataFrame a encoder.
        user_encoder: Encodeur des utilisateurs.
        item_encoder: Encodeur des items.
        handle_unknown: Strategie pour les IDs inconnus ('ignore' ou 'error').

    Returns:
        DataFrame avec colonnes user_idx et item_idx.
    """
    df = df.copy()

    # Filtrer les utilisateurs et items inconnus si necessaire
    if handle_unknown == "ignore":
        known_users = set(user_encoder.classes_)
        known_items = set(item_encoder.classes_)

        initial_count = len(df)
        df = df[
            df["user_id"].isin(known_users) &
            df["item_id"].isin(known_items)
        ]
        removed = initial_count - len(df)

        if removed > 0:
            logger.warning(
                "Interactions avec IDs inconnus ignorees",
                removed=removed,
            )

    # Encoder les IDs
    df["user_idx"] = user_encoder.transform(df["user_id"])
    df["item_idx"] = item_encoder.transform(df["item_id"])

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
    Point d'entree principal pour la division du dataset.

    Cette fonction est appelee par DVC pour diviser le dataset
    selon les parametres definis dans params.yaml.
    """
    # Charger les parametres
    params = load_params()
    data_params = params.get("data", {})
    split_params = params.get("split", {})

    interim_dir = PROJECT_ROOT / data_params.get("interim_dir", "data/interim")
    processed_dir = PROJECT_ROOT / data_params.get("processed_dir", "data/processed")

    train_ratio = split_params.get("train_ratio", 0.8)
    val_ratio = split_params.get("val_ratio", 0.1)
    test_ratio = split_params.get("test_ratio", 0.1)
    temporal = split_params.get("temporal_split", True)
    min_user_interactions = split_params.get("min_user_interactions", 5)
    min_item_interactions = split_params.get("min_item_interactions", 5)

    logger.info(
        "Demarrage de la division du dataset",
        temporal_split=temporal,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Charger le dataset
    interactions_df = load_parquet(interim_dir / "interactions.parquet")

    # Filtrer le cold-start
    interactions_df = filter_cold_start(
        interactions_df,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    # Diviser le dataset
    if temporal:
        train_df, val_df, test_df = temporal_split(
            interactions_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        train_df, val_df, test_df = random_split(
            interactions_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    # Creer les encodeurs a partir de l'entrainement
    user_encoder, item_encoder = create_encoders(train_df)

    # Encoder les IDs
    train_df = encode_ids(train_df, user_encoder, item_encoder)
    val_df = encode_ids(val_df, user_encoder, item_encoder, handle_unknown="ignore")
    test_df = encode_ids(test_df, user_encoder, item_encoder, handle_unknown="ignore")

    # Sauvegarder les datasets
    save_parquet(train_df, processed_dir / "train.parquet")
    save_parquet(val_df, processed_dir / "val.parquet")
    save_parquet(test_df, processed_dir / "test.parquet")

    # Sauvegarder les encodeurs
    save_joblib(user_encoder, processed_dir / "user_encoder.joblib")
    save_joblib(item_encoder, processed_dir / "item_encoder.joblib")

    logger.info(
        "Division du dataset terminee",
        train_path=str(processed_dir / "train.parquet"),
        val_path=str(processed_dir / "val.parquet"),
        test_path=str(processed_dir / "test.parquet"),
    )


if __name__ == "__main__":
    main()
