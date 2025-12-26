# -*- coding: utf-8 -*-
"""
API FastAPI pour le systeme de recommandation.

Cette API expose des endpoints pour generer des recommandations
en temps reel a partir du modele entraine.

Endpoints:
- GET /health: Verification de l'etat de sante
- POST /recommend: Recommandations pour un utilisateur
- POST /similar-items: Items similaires a un item donne
- GET /users: Liste des utilisateurs disponibles
- GET /items: Liste des items disponibles
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.recommend import Recommender
from src.utils.logging import configure_logging, get_logger

# Configuration du logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_format = os.getenv("LOG_FORMAT", "json") == "json"
configure_logging(level=log_level, json_format=log_format)

logger = get_logger(__name__)

# Instance globale du recommandeur
recommender: Optional[Recommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.

    Charge le modele au demarrage et le libere a l'arret.
    """
    global recommender

    logger.info("Demarrage de l'API de recommandation")

    # Charger le recommandeur
    try:
        recommender = Recommender().load()
        logger.info(
            "Recommandeur charge avec succes",
            n_users=len(recommender.get_all_users()),
            n_items=len(recommender.get_all_items()),
        )
    except Exception as e:
        logger.error("Erreur lors du chargement du recommandeur", error=str(e))
        recommender = None

    yield

    # Nettoyage
    logger.info("Arret de l'API de recommandation")
    recommender = None


# Creation de l'application FastAPI
app = FastAPI(
    title="MLOps Recommender API",
    description="API de recommandation de films basee sur le jeu de donnees MovieLens",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Modeles Pydantic
# ============================================================================


class HealthResponse(BaseModel):
    """Reponse du endpoint de sante."""

    status: str = Field(..., description="Etat de sante de l'API")
    timestamp: str = Field(..., description="Horodatage de la verification")
    model_loaded: bool = Field(..., description="Indique si le modele est charge")
    n_users: Optional[int] = Field(None, description="Nombre d'utilisateurs")
    n_items: Optional[int] = Field(None, description="Nombre d'items")


class RecommendRequest(BaseModel):
    """Requete pour obtenir des recommandations."""

    user_id: int = Field(..., description="ID de l'utilisateur")
    k: int = Field(default=10, ge=1, le=100, description="Nombre de recommandations")
    exclude_seen: bool = Field(default=True, description="Exclure les items deja vus")


class RecommendationItem(BaseModel):
    """Un item recommande."""

    item_id: int = Field(..., description="ID de l'item")
    score: float = Field(..., description="Score de recommandation")
    rank: int = Field(..., description="Rang de la recommandation")
    title: Optional[str] = Field(None, description="Titre de l'item")
    genres: Optional[str] = Field(None, description="Genres de l'item")


class RecommendResponse(BaseModel):
    """Reponse du endpoint de recommandation."""

    user_id: int = Field(..., description="ID de l'utilisateur")
    recommendations: List[RecommendationItem] = Field(
        ..., description="Liste des recommandations"
    )
    timestamp: str = Field(..., description="Horodatage de la requete")


class SimilarItemsRequest(BaseModel):
    """Requete pour obtenir des items similaires."""

    item_id: int = Field(..., description="ID de l'item de reference")
    k: int = Field(default=10, ge=1, le=100, description="Nombre d'items similaires")


class SimilarItem(BaseModel):
    """Un item similaire."""

    item_id: int = Field(..., description="ID de l'item")
    similarity: float = Field(..., description="Score de similarite")
    rank: int = Field(..., description="Rang de similarite")
    title: Optional[str] = Field(None, description="Titre de l'item")
    genres: Optional[str] = Field(None, description="Genres de l'item")


class SimilarItemsResponse(BaseModel):
    """Reponse du endpoint d'items similaires."""

    item_id: int = Field(..., description="ID de l'item de reference")
    similar_items: List[SimilarItem] = Field(..., description="Liste des items similaires")
    timestamp: str = Field(..., description="Horodatage de la requete")


class UserListResponse(BaseModel):
    """Reponse contenant la liste des utilisateurs."""

    users: List[int] = Field(..., description="Liste des IDs utilisateurs")
    count: int = Field(..., description="Nombre total d'utilisateurs")


class ItemListResponse(BaseModel):
    """Reponse contenant la liste des items."""

    items: List[int] = Field(..., description="Liste des IDs items")
    count: int = Field(..., description="Nombre total d'items")


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Verifie l'etat de sante de l'API.

    Retourne le statut de l'API et des informations sur le modele charge.
    """
    model_loaded = recommender is not None
    n_users = None
    n_items = None

    if model_loaded and recommender is not None:
        try:
            n_users = len(recommender.get_all_users())
            n_items = len(recommender.get_all_items())
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        n_users=n_users,
        n_items=n_items,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendRequest) -> RecommendResponse:
    """
    Genere des recommandations pour un utilisateur.

    Args:
        request: Requete contenant l'ID utilisateur et les parametres.

    Returns:
        Liste des items recommandes avec leurs scores.

    Raises:
        HTTPException: Si le modele n'est pas charge ou en cas d'erreur.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Le modele de recommandation n'est pas disponible",
        )

    logger.info(
        "Requete de recommandation",
        user_id=request.user_id,
        k=request.k,
    )

    try:
        recommendations = recommender.recommend(
            user_id=request.user_id,
            n=request.k,
            exclude_seen=request.exclude_seen,
            include_metadata=True,
        )

        items = [
            RecommendationItem(
                item_id=rec["item_id"],
                score=rec["score"],
                rank=rec["rank"],
                title=rec.get("title"),
                genres=rec.get("genres"),
            )
            for rec in recommendations
        ]

        return RecommendResponse(
            user_id=request.user_id,
            recommendations=items,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(
            "Erreur lors de la generation des recommandations",
            user_id=request.user_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la generation des recommandations: {str(e)}",
        )


@app.post("/similar-items", response_model=SimilarItemsResponse, tags=["Recommendations"])
async def get_similar_items(request: SimilarItemsRequest) -> SimilarItemsResponse:
    """
    Trouve les items similaires a un item donne.

    Args:
        request: Requete contenant l'ID de l'item et les parametres.

    Returns:
        Liste des items similaires avec leurs scores de similarite.

    Raises:
        HTTPException: Si le modele n'est pas charge ou en cas d'erreur.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Le modele de recommandation n'est pas disponible",
        )

    logger.info(
        "Requete d'items similaires",
        item_id=request.item_id,
        k=request.k,
    )

    try:
        similar = recommender.similar_items(
            item_id=request.item_id,
            n=request.k,
            include_metadata=True,
        )

        items = [
            SimilarItem(
                item_id=item["item_id"],
                similarity=item["similarity"],
                rank=item["rank"],
                title=item.get("title"),
                genres=item.get("genres"),
            )
            for item in similar
        ]

        return SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=items,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(
            "Erreur lors de la recherche d'items similaires",
            item_id=request.item_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recherche d'items similaires: {str(e)}",
        )


@app.get("/users", response_model=UserListResponse, tags=["Data"])
async def list_users(
    limit: int = Query(default=100, ge=1, le=1000, description="Nombre max d'utilisateurs"),
    offset: int = Query(default=0, ge=0, description="Offset pour la pagination"),
) -> UserListResponse:
    """
    Liste les utilisateurs disponibles.

    Args:
        limit: Nombre maximum d'utilisateurs a retourner.
        offset: Offset pour la pagination.

    Returns:
        Liste des IDs utilisateurs.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Le modele de recommandation n'est pas disponible",
        )

    all_users = recommender.get_all_users()
    paginated = all_users[offset : offset + limit]

    return UserListResponse(
        users=paginated,
        count=len(all_users),
    )


@app.get("/items", response_model=ItemListResponse, tags=["Data"])
async def list_items(
    limit: int = Query(default=100, ge=1, le=1000, description="Nombre max d'items"),
    offset: int = Query(default=0, ge=0, description="Offset pour la pagination"),
) -> ItemListResponse:
    """
    Liste les items disponibles.

    Args:
        limit: Nombre maximum d'items a retourner.
        offset: Offset pour la pagination.

    Returns:
        Liste des IDs items.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Le modele de recommandation n'est pas disponible",
        )

    all_items = recommender.get_all_items()
    paginated = all_items[offset : offset + limit]

    return ItemListResponse(
        items=paginated,
        count=len(all_items),
    )


@app.get("/user/{user_id}/history", tags=["Data"])
async def get_user_history(user_id: int) -> Dict[str, Any]:
    """
    Retourne l'historique d'un utilisateur.

    Args:
        user_id: ID de l'utilisateur.

    Returns:
        Historique des interactions de l'utilisateur.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Le modele de recommandation n'est pas disponible",
        )

    history = recommender.get_user_history(user_id)

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur {user_id} non trouve ou sans historique",
        )

    return {
        "user_id": user_id,
        "history": history,
        "count": len(history),
    }


# Point d'entree pour uvicorn
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "src.serving.api:app",
        host=host,
        port=port,
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
