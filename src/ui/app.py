# -*- coding: utf-8 -*-
"""
Interface Streamlit pour le systeme de recommandation.

Cette application web permet aux utilisateurs d'explorer les recommandations
de films, de visualiser les items similaires et d'interagir avec le modele.
"""

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Film Recommender",
    page_icon="[icon:film]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URL de l'API
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ============================================================================
# Fonctions utilitaires
# ============================================================================


@st.cache_data(ttl=60)
def get_api_health() -> Optional[Dict[str, Any]]:
    """
    Verifie l'etat de sante de l'API.

    Returns:
        Dictionnaire avec les informations de sante, ou None si erreur.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


@st.cache_data(ttl=300)
def get_users(limit: int = 100) -> List[int]:
    """
    Recupere la liste des utilisateurs depuis l'API.

    Args:
        limit: Nombre maximum d'utilisateurs.

    Returns:
        Liste des IDs utilisateurs.
    """
    try:
        response = requests.get(
            f"{API_URL}/users",
            params={"limit": limit},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("users", [])
    except requests.exceptions.RequestException:
        pass
    return []


@st.cache_data(ttl=300)
def get_items(limit: int = 100) -> List[int]:
    """
    Recupere la liste des items depuis l'API.

    Args:
        limit: Nombre maximum d'items.

    Returns:
        Liste des IDs items.
    """
    try:
        response = requests.get(
            f"{API_URL}/items",
            params={"limit": limit},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("items", [])
    except requests.exceptions.RequestException:
        pass
    return []


def get_recommendations(
    user_id: int,
    k: int = 10,
    exclude_seen: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Obtient les recommandations pour un utilisateur.

    Args:
        user_id: ID de l'utilisateur.
        k: Nombre de recommandations.
        exclude_seen: Exclure les items deja vus.

    Returns:
        Dictionnaire avec les recommandations, ou None si erreur.
    """
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={
                "user_id": user_id,
                "k": k,
                "exclude_seen": exclude_seen,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion a l'API: {e}")
    return None


def get_similar_items(item_id: int, k: int = 10) -> Optional[Dict[str, Any]]:
    """
    Obtient les items similaires a un item donne.

    Args:
        item_id: ID de l'item.
        k: Nombre d'items similaires.

    Returns:
        Dictionnaire avec les items similaires, ou None si erreur.
    """
    try:
        response = requests.post(
            f"{API_URL}/similar-items",
            json={
                "item_id": item_id,
                "k": k,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion a l'API: {e}")
    return None


def get_user_history(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Recupere l'historique d'un utilisateur.

    Args:
        user_id: ID de l'utilisateur.

    Returns:
        Dictionnaire avec l'historique, ou None si erreur.
    """
    try:
        response = requests.get(
            f"{API_URL}/user/{user_id}/history",
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


# ============================================================================
# Interface utilisateur
# ============================================================================


def render_sidebar() -> Dict[str, Any]:
    """
    Affiche la barre laterale avec les controles.

    Returns:
        Dictionnaire avec les parametres selectionnes.
    """
    st.sidebar.title("[icon:settings] Configuration")

    # Verification de l'API
    health = get_api_health()

    if health and health.get("status") == "healthy":
        st.sidebar.success(f"[icon:check_circle] API connectee")
        st.sidebar.caption(
            f"Utilisateurs: {health.get('n_users', 'N/A')} | "
            f"Items: {health.get('n_items', 'N/A')}"
        )
    else:
        st.sidebar.error("[icon:error] API non disponible")
        st.sidebar.caption(f"URL: {API_URL}")

    st.sidebar.divider()

    # Selection de la page
    page = st.sidebar.radio(
        "Navigation",
        options=["Recommandations", "Items similaires", "Historique"],
        index=0,
    )

    st.sidebar.divider()

    # Nombre de resultats
    k = st.sidebar.slider(
        "Nombre de resultats",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    return {
        "page": page,
        "k": k,
        "api_healthy": health is not None and health.get("status") == "healthy",
    }


def render_recommendations_page(k: int) -> None:
    """
    Affiche la page de recommandations.

    Args:
        k: Nombre de recommandations a afficher.
    """
    st.title("[icon:movie] Recommandations personnalisees")
    st.markdown("Selectionnez un utilisateur pour obtenir des recommandations de films.")

    # Selection de l'utilisateur
    users = get_users(limit=500)

    if not users:
        st.warning("Impossible de charger la liste des utilisateurs.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_user = st.selectbox(
            "Selectionner un utilisateur",
            options=users,
            format_func=lambda x: f"Utilisateur {x}",
        )

    with col2:
        exclude_seen = st.checkbox("Exclure les films deja vus", value=True)

    if st.button("Obtenir les recommandations", type="primary"):
        with st.spinner("Chargement des recommandations..."):
            result = get_recommendations(selected_user, k, exclude_seen)

            if result and result.get("recommendations"):
                st.success(
                    f"[icon:thumb_up] {len(result['recommendations'])} recommandations "
                    f"pour l'utilisateur {selected_user}"
                )

                # Afficher les recommandations
                for rec in result["recommendations"]:
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 2])

                        with col1:
                            st.markdown(f"### #{rec['rank']}")

                        with col2:
                            title = rec.get("title", f"Film {rec['item_id']}")
                            genres = rec.get("genres", "")
                            st.markdown(f"**{title}**")
                            if genres:
                                st.caption(f"[icon:category] {genres}")

                        with col3:
                            score = rec["score"]
                            st.metric("Score", f"{score:.3f}")

                        st.divider()
            else:
                st.info("Aucune recommandation trouvee pour cet utilisateur.")


def render_similar_items_page(k: int) -> None:
    """
    Affiche la page d'items similaires.

    Args:
        k: Nombre d'items similaires a afficher.
    """
    st.title("[icon:compare_arrows] Items similaires")
    st.markdown("Selectionnez un film pour trouver des films similaires.")

    # Selection de l'item
    items = get_items(limit=500)

    if not items:
        st.warning("Impossible de charger la liste des items.")
        return

    selected_item = st.selectbox(
        "Selectionner un film",
        options=items,
        format_func=lambda x: f"Film {x}",
    )

    if st.button("Trouver des films similaires", type="primary"):
        with st.spinner("Recherche des films similaires..."):
            result = get_similar_items(selected_item, k)

            if result and result.get("similar_items"):
                st.success(
                    f"[icon:thumb_up] {len(result['similar_items'])} films similaires "
                    f"au film {selected_item}"
                )

                # Afficher les items similaires
                for item in result["similar_items"]:
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 2])

                        with col1:
                            st.markdown(f"### #{item['rank']}")

                        with col2:
                            title = item.get("title", f"Film {item['item_id']}")
                            genres = item.get("genres", "")
                            st.markdown(f"**{title}**")
                            if genres:
                                st.caption(f"[icon:category] {genres}")

                        with col3:
                            similarity = item["similarity"]
                            st.metric("Similarite", f"{similarity:.3f}")

                        st.divider()
            else:
                st.info("Aucun film similaire trouve.")


def render_history_page() -> None:
    """Affiche la page d'historique utilisateur."""
    st.title("[icon:history] Historique utilisateur")
    st.markdown("Consultez l'historique des interactions d'un utilisateur.")

    # Selection de l'utilisateur
    users = get_users(limit=500)

    if not users:
        st.warning("Impossible de charger la liste des utilisateurs.")
        return

    selected_user = st.selectbox(
        "Selectionner un utilisateur",
        options=users,
        format_func=lambda x: f"Utilisateur {x}",
    )

    if st.button("Afficher l'historique", type="primary"):
        with st.spinner("Chargement de l'historique..."):
            result = get_user_history(selected_user)

            if result and result.get("history"):
                st.success(
                    f"[icon:check] {result['count']} interactions "
                    f"pour l'utilisateur {selected_user}"
                )

                # Afficher l'historique
                for item in result["history"]:
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 2])

                        with col1:
                            st.markdown(f"**ID: {item['item_id']}**")

                        with col2:
                            title = item.get("title", f"Film {item['item_id']}")
                            st.markdown(f"{title}")

                        with col3:
                            rating = item.get("rating", 0)
                            stars = "[icon:star]" * int(rating)
                            st.markdown(f"{stars} ({rating:.1f})")

                        st.divider()
            else:
                st.info("Aucun historique trouve pour cet utilisateur.")


def main() -> None:
    """Point d'entree principal de l'application Streamlit."""
    # Styles CSS personnalises
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    params = render_sidebar()

    # Contenu principal
    if not params["api_healthy"]:
        st.error(
            "[icon:warning] L'API de recommandation n'est pas disponible. "
            "Veuillez verifier que le service est demarre."
        )
        st.code(f"API_URL = {API_URL}")
        st.info("Lancez l'API avec: `make serve` ou `uvicorn src.serving.api:app`")
        return

    # Navigation
    if params["page"] == "Recommandations":
        render_recommendations_page(params["k"])
    elif params["page"] == "Items similaires":
        render_similar_items_page(params["k"])
    elif params["page"] == "Historique":
        render_history_page()


if __name__ == "__main__":
    main()
