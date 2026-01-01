# -*- coding: utf-8 -*-
"""
Application de Recommandation de Films

Interface moderne et intuitive pour explorer les recommandations
personnalisees basees sur l'intelligence artificielle.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Recommandation Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",  # Sidebar cachee par defaut pour plus d'espace
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
CURRENT_DIR = Path(__file__).parent


def load_css() -> None:
    """Charge les styles CSS."""
    css_path = CURRENT_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# =============================================================================
# API
# =============================================================================

@st.cache_data(ttl=60)
def api_health() -> Optional[Dict]:
    """Verifie l'etat de l'API."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def api_users() -> List[int]:
    """Liste des utilisateurs."""
    try:
        r = requests.get(f"{API_URL}/users", params={"limit": 500}, timeout=10)
        return r.json().get("users", []) if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=300)
def api_items() -> List[int]:
    """Liste des films."""
    try:
        r = requests.get(f"{API_URL}/items", params={"limit": 500}, timeout=10)
        return r.json().get("items", []) if r.ok else []
    except Exception:
        return []


def api_recommend(user_id: int, k: int, exclude: bool) -> Optional[Dict]:
    """Genere des recommandations."""
    try:
        r = requests.post(
            f"{API_URL}/recommend",
            json={"user_id": user_id, "k": k, "exclude_seen": exclude},
            timeout=15,
        )
        return r.json() if r.ok else None
    except Exception:
        return None


def api_similar(item_id: int, k: int) -> Optional[Dict]:
    """Trouve des films similaires."""
    try:
        r = requests.post(
            f"{API_URL}/similar-items",
            json={"item_id": item_id, "k": k},
            timeout=15,
        )
        return r.json() if r.ok else None
    except Exception:
        return None


def api_history(user_id: int) -> Optional[Dict]:
    """Historique utilisateur."""
    try:
        r = requests.get(f"{API_URL}/user/{user_id}/history", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


# =============================================================================
# Composants UI
# =============================================================================

def header() -> None:
    """Affiche l'en-tete de l'application."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Recommandation de Films")
        st.caption("Decouvrez vos prochains films preferes grace a l'IA")
    
    with col2:
        health = api_health()
        if health and health.get("status") == "healthy":
            st.success(f"Connecte - {health.get('n_items', 0)} films")
        else:
            st.error("API deconnectee")


def tab_recommendations() -> None:
    """Onglet Recommandations."""
    st.markdown("### Recommandations Personnalisees")
    st.markdown("Selectionnez votre profil pour obtenir des suggestions adaptees.")
    
    users = api_users()
    if not users:
        st.warning("Impossible de charger les utilisateurs.")
        return
    
    # Formulaire compact
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            user_id = st.selectbox(
                "Votre profil",
                users,
                format_func=lambda x: f"Utilisateur {x}",
                label_visibility="collapsed",
                key="rec_user"
            )
        
        with col2:
            k = st.select_slider(
                "Nombre",
                options=[5, 10, 15, 20, 30],
                value=10,
                label_visibility="collapsed",
                key="rec_k"
            )
        
        with col3:
            exclude = st.checkbox("Nouveaux uniquement", value=True, key="rec_exclude")
    
    st.markdown("")  # Espacement
    
    if st.button("Obtenir mes recommandations", type="primary", use_container_width=True, key="rec_btn"):
        with st.spinner("Analyse de votre profil..."):
            data = api_recommend(user_id, k, exclude)
        
        if data and data.get("recommendations"):
            recs = data["recommendations"]
            st.success(f"{len(recs)} films recommandes pour vous")
            
            # Affichage en grille de cartes
            for i in range(0, len(recs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(recs):
                        rec = recs[i + j]
                        with col:
                            with st.container():
                                title = rec.get("title") or f"Film {rec['item_id']}"
                                st.markdown(f"**{title}**")
                                st.caption(rec.get("genres", ""))
                                st.progress(min(rec["score"], 1.0))
                                st.caption(f"Pertinence: {rec['score']:.1%}")
        else:
            st.info("Aucune recommandation disponible pour ce profil.")


def tab_discover() -> None:
    """Onglet Decouvrir (films similaires)."""
    st.markdown("### Explorer des Films Similaires")
    st.markdown("Choisissez un film que vous aimez pour en decouvrir de similaires.")
    
    items = api_items()
    if not items:
        st.warning("Catalogue indisponible.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        item_id = st.selectbox(
            "Film de reference",
            items,
            format_func=lambda x: f"Film {x}",
            label_visibility="collapsed",
            key="sim_item"
        )
    
    with col2:
        k = st.select_slider(
            "Nombre",
            options=[5, 10, 15, 20],
            value=10,
            label_visibility="collapsed",
            key="sim_k"
        )
    
    st.markdown("")
    
    if st.button("Trouver des films similaires", type="primary", use_container_width=True, key="sim_btn"):
        with st.spinner("Recherche en cours..."):
            data = api_similar(item_id, k)
        
        if data and data.get("similar_items"):
            sims = data["similar_items"]
            st.success(f"{len(sims)} films similaires trouves")
            
            df = pd.DataFrame([
                {
                    "Titre": s.get("title", f"Film {s['item_id']}"),
                    "Genre": s.get("genres", "-"),
                    "Similarite": s["similarity"],
                }
                for s in sims
            ])
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Similarite": st.column_config.ProgressColumn(
                        format="%.0f%%",
                        min_value=0,
                        max_value=1,
                    ),
                },
            )
        else:
            st.info("Aucun film similaire trouve.")


def tab_history() -> None:
    """Onglet Historique."""
    st.markdown("### Votre Historique")
    st.markdown("Consultez les films que vous avez notes.")
    
    users = api_users()
    if not users:
        st.warning("Profils indisponibles.")
        return
    
    user_id = st.selectbox(
        "Profil",
        users,
        format_func=lambda x: f"Utilisateur {x}",
        label_visibility="collapsed",
        key="hist_user"
    )
    
    st.markdown("")
    
    if st.button("Voir mon historique", type="primary", use_container_width=True, key="hist_btn"):
        with st.spinner("Chargement..."):
            data = api_history(user_id)
        
        if data and data.get("history"):
            hist = data["history"]
            st.success(f"{data['count']} films dans votre historique")
            
            df = pd.DataFrame([
                {
                    "Titre": h.get("title", f"Film {h['item_id']}"),
                    "Note": h.get("rating", 0),
                }
                for h in hist
            ])
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Note": st.column_config.NumberColumn(
                        format="%.1f / 5",
                    ),
                },
            )
        else:
            st.info("Historique vide.")


def tab_stats() -> None:
    """Onglet Statistiques."""
    st.markdown("### Statistiques du Systeme")
    
    health = api_health()
    
    if health:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Utilisateurs", health.get("n_users", "N/A"))
        
        with col2:
            st.metric("Films", health.get("n_items", "N/A"))
        
        with col3:
            status = "En ligne" if health.get("status") == "healthy" else "Degrade"
            st.metric("Statut", status)
    else:
        st.error("Impossible de recuperer les statistiques.")


# =============================================================================
# Application principale
# =============================================================================

def main() -> None:
    """Point d'entree."""
    load_css()
    
    # Verification API
    health = api_health()
    if not health or health.get("status") != "healthy":
        st.error("Le service de recommandation n'est pas disponible.")
        st.info(f"Verifiez que l'API est demarree sur {API_URL}")
        return
    
    # En-tete
    header()
    
    st.markdown("---")
    
    # Navigation par onglets
    tabs = st.tabs([
        "Recommandations",
        "Decouvrir",
        "Historique",
        "Statistiques"
    ])
    
    with tabs[0]:
        tab_recommendations()
    
    with tabs[1]:
        tab_discover()
    
    with tabs[2]:
        tab_history()
    
    with tabs[3]:
        tab_stats()


if __name__ == "__main__":
    main()
