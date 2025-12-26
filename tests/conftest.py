# -*- coding: utf-8 -*-
"""
Configuration pytest pour le projet.

Ce fichier configure les fixtures et options globales pour les tests.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure l'environnement de test."""
    import os

    # Definir les variables d'environnement pour les tests
    os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["LOG_FORMAT"] = "console"

    yield

    # Nettoyage apres les tests
    pass
