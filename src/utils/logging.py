# -*- coding: utf-8 -*-
"""
Module de configuration de la journalisation pour le systeme de recommandation.

Ce module configure structlog pour une journalisation structuree en JSON,
facilitant l'analyse des logs en production et le debogage en developpement.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import Processor


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    include_timestamp: bool = True,
) -> None:
    """
    Configure la journalisation structuree pour l'application.

    Cette fonction initialise structlog avec les processeurs appropries
    pour produire des logs structures en JSON (production) ou formattes
    pour la console (developpement).

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Utiliser le format JSON. Par defaut: True.
        include_timestamp: Inclure l'horodatage. Par defaut: True.
    """
    # Configurer le niveau de logging standard
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Definir les processeurs structlog
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamp:
        shared_processors.insert(
            0,
            structlog.processors.TimeStamper(fmt="iso"),
        )

    if json_format:
        # Format JSON pour la production
        shared_processors.append(
            structlog.processors.format_exc_info,
        )
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Format console pour le developpement
        shared_processors.append(
            structlog.dev.set_exc_info,
        )
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Obtient un logger configure pour le module specifie.

    Cette fonction retourne un logger structlog configure selon
    les parametres definis par configure_logging().

    Args:
        name: Nom du module pour le logger. Si None, utilise le nom du module appelant.

    Returns:
        Logger structlog configure.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Demarrage du traitement", user_id=123)
    """
    return structlog.get_logger(name)


def log_with_context(
    logger: structlog.stdlib.BoundLogger,
    level: str,
    message: str,
    **context: Any,
) -> None:
    """
    Log un message avec un contexte supplementaire.

    Cette fonction utilitaire simplifie l'ajout de contexte aux logs,
    garantissant une structure coherente des messages.

    Args:
        logger: Logger structlog a utiliser.
        level: Niveau de log (info, warning, error, etc.).
        message: Message a logger.
        **context: Contexte supplementaire a inclure dans le log.
    """
    log_method = getattr(logger, level.lower())
    log_method(message, **context)


class LogContext:
    """
    Gestionnaire de contexte pour ajouter des informations aux logs.

    Cette classe permet d'ajouter temporairement des informations de contexte
    a tous les logs emis dans le bloc with, facilitant le suivi des operations.

    Example:
        >>> with LogContext(request_id="abc123", user_id=42):
        ...     logger.info("Traitement en cours")
        ...     # Le log inclura request_id et user_id
    """

    def __init__(self, **context: Any) -> None:
        """
        Initialise le contexte de log.

        Args:
            **context: Paires cle-valeur a ajouter au contexte de log.
        """
        self._context = context
        self._token: Optional[object] = None

    def __enter__(self) -> "LogContext":
        """Entre dans le contexte et lie les variables."""
        self._token = structlog.contextvars.bind_contextvars(**self._context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Quitte le contexte et nettoie les variables."""
        structlog.contextvars.unbind_contextvars(*self._context.keys())


def log_function_call(
    logger: structlog.stdlib.BoundLogger,
    func_name: str,
    **kwargs: Any,
) -> None:
    """
    Log l'appel d'une fonction avec ses arguments.

    Utile pour le debogage et le suivi des appels de fonctions critiques.

    Args:
        logger: Logger structlog a utiliser.
        func_name: Nom de la fonction appelee.
        **kwargs: Arguments de la fonction a logger.
    """
    # Filtrer les arguments sensibles ou trop volumineux
    safe_kwargs: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if "password" in key.lower() or "secret" in key.lower():
            safe_kwargs[key] = "***"
        elif isinstance(value, (list, dict)) and len(str(value)) > 200:
            safe_kwargs[key] = f"<{type(value).__name__} len={len(value)}>"
        else:
            safe_kwargs[key] = value

    logger.debug(
        "Appel de fonction",
        function=func_name,
        arguments=safe_kwargs,
    )


# Configurer le logging par defaut au chargement du module
configure_logging()
