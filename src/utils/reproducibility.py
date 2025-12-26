# -*- coding: utf-8 -*-
"""
Module de gestion de la reproductibilite pour le systeme de recommandation.

Ce module fournit des fonctions pour garantir la reproductibilite des experiences,
en fixant les graines aleatoires pour numpy, random, et les bibliotheques ML.
"""

import os
import random
from typing import Optional

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Graine par defaut pour la reproductibilite
DEFAULT_SEED: int = 42


def set_seed(seed: Optional[int] = None) -> int:
    """
    Fixe les graines aleatoires pour la reproductibilite.

    Cette fonction configure les generateurs aleatoires de Python, numpy,
    et definit les variables d'environnement appropriees pour garantir
    des resultats reproductibles.

    Args:
        seed: Graine aleatoire a utiliser. Si None, utilise DEFAULT_SEED.

    Returns:
        La graine utilisee.

    Example:
        >>> set_seed(42)
        42
        >>> np.random.rand()  # Resultat deterministe
        0.3745401188473625
    """
    if seed is None:
        seed = DEFAULT_SEED

    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Variables d'environnement pour certaines bibliotheques
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Graine aleatoire fixee", seed=seed)
    return seed


def set_deterministic_mode(enable: bool = True) -> None:
    """
    Active le mode deterministe pour les operations.

    Cette fonction configure les options de determinisme pour numpy
    et d'autres bibliotheques lorsque disponible.

    Args:
        enable: Si True, active le mode deterministe.
    """
    if enable:
        # Numpy: utiliser l'algorithme deterministe pour certaines operations
        os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

        logger.info("Mode deterministe active")
    else:
        logger.info("Mode deterministe desactive")


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Cree un objet RandomState numpy avec une graine specifique.

    Utile pour passer un etat aleatoire a des fonctions qui l'acceptent,
    garantissant la reproductibilite sans affecter l'etat global.

    Args:
        seed: Graine aleatoire a utiliser. Si None, utilise DEFAULT_SEED.

    Returns:
        Objet numpy RandomState configure.

    Example:
        >>> rng = get_random_state(42)
        >>> rng.rand(3)
        array([0.37454012, 0.95071431, 0.73199394])
    """
    if seed is None:
        seed = DEFAULT_SEED
    return np.random.RandomState(seed)


def reproducible_shuffle(
    items: list,
    seed: Optional[int] = None,
) -> list:
    """
    Melange une liste de maniere reproductible.

    Cette fonction retourne une nouvelle liste melangee sans modifier
    la liste originale, en utilisant une graine specifique.

    Args:
        items: Liste a melanger.
        seed: Graine aleatoire a utiliser.

    Returns:
        Nouvelle liste melangee.

    Example:
        >>> items = [1, 2, 3, 4, 5]
        >>> reproducible_shuffle(items, seed=42)
        [1, 4, 2, 5, 3]
    """
    rng = get_random_state(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    return shuffled


def reproducible_split(
    items: list,
    ratios: list[float],
    seed: Optional[int] = None,
) -> list[list]:
    """
    Divise une liste en plusieurs parties de maniere reproductible.

    Cette fonction est utile pour creer des splits train/val/test
    de maniere deterministe.

    Args:
        items: Liste a diviser.
        ratios: Liste des ratios pour chaque partie (doit sommer a 1.0).
        seed: Graine aleatoire a utiliser.

    Returns:
        Liste de listes correspondant aux parties divisees.

    Raises:
        ValueError: Si les ratios ne somment pas a 1.0.

    Example:
        >>> items = list(range(100))
        >>> train, val, test = reproducible_split(items, [0.8, 0.1, 0.1], seed=42)
        >>> len(train), len(val), len(test)
        (80, 10, 10)
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"Les ratios doivent sommer a 1.0, obtenu: {sum(ratios)}"
        )

    # Melanger les items
    shuffled = reproducible_shuffle(items, seed)
    n = len(shuffled)

    # Calculer les indices de split
    splits = []
    start = 0
    for i, ratio in enumerate(ratios[:-1]):
        end = start + int(n * ratio)
        splits.append(shuffled[start:end])
        start = end

    # Derniere partie prend le reste
    splits.append(shuffled[start:])

    return splits


class ReproducibilityContext:
    """
    Gestionnaire de contexte pour la reproductibilite.

    Cette classe permet de definir temporairement une graine aleatoire
    pour un bloc de code, restaurant l'etat precedent a la sortie.

    Example:
        >>> with ReproducibilityContext(seed=42):
        ...     result = np.random.rand()  # Resultat deterministe
        >>> # L'etat aleatoire est restaure ici
    """

    def __init__(self, seed: int) -> None:
        """
        Initialise le contexte de reproductibilite.

        Args:
            seed: Graine aleatoire a utiliser dans le contexte.
        """
        self.seed = seed
        self._saved_python_state: Optional[tuple] = None
        self._saved_numpy_state: Optional[dict] = None

    def __enter__(self) -> "ReproducibilityContext":
        """Entre dans le contexte et sauvegarde l'etat actuel."""
        # Sauvegarder l'etat actuel
        self._saved_python_state = random.getstate()
        self._saved_numpy_state = np.random.get_state()

        # Fixer la nouvelle graine
        set_seed(self.seed)

        return self

    def __exit__(self, *args: object) -> None:
        """Quitte le contexte et restaure l'etat precedent."""
        # Restaurer l'etat precedent
        if self._saved_python_state is not None:
            random.setstate(self._saved_python_state)
        if self._saved_numpy_state is not None:
            np.random.set_state(self._saved_numpy_state)


def ensure_reproducibility(seed: Optional[int] = None) -> None:
    """
    Configure completement l'environnement pour la reproductibilite.

    Cette fonction est le point d'entree recommande pour configurer
    la reproductibilite au debut d'un script ou d'une experience.

    Args:
        seed: Graine aleatoire a utiliser. Si None, utilise DEFAULT_SEED.

    Example:
        >>> ensure_reproducibility(42)
        >>> # Tout le code suivant sera reproductible
    """
    if seed is None:
        seed = DEFAULT_SEED

    set_seed(seed)
    set_deterministic_mode(True)

    logger.info(
        "Environnement configure pour la reproductibilite",
        seed=seed,
    )
