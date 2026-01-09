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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
CURRENT_DIR = Path(__file__).parent


def load_css() -> None:
    """Charge les styles CSS et FontAwesome."""
    # FontAwesome 6.4.0 CDN
    st.markdown('''
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    ''', unsafe_allow_html=True)
    
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


@st.cache_data(ttl=300)
def api_items_with_metadata() -> List[Dict]:
    """Liste des films avec titres et genres."""
    try:
        r = requests.get(f"{API_URL}/items-with-metadata", params={"limit": 500}, timeout=10)
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
# Helper Functions
# =============================================================================

def get_score_label(score: float) -> str:
    """Convert score to qualitative label."""
    if score >= 0.8:
        return "Tres recommande"
    elif score >= 0.6:
        return "Recommande"
    else:
        return "A decouvrir"


def get_score_icon(score: float) -> str:
    """Get icon based on score."""
    if score >= 0.8:
        return "fa-star"
    elif score >= 0.6:
        return "fa-thumbs-up"
    else:
        return "fa-lightbulb"


# =============================================================================
# UI Components
# =============================================================================

def header() -> None:
    """Affiche l'en-tete minimal et elegant."""
    health = api_health()
    n_films = health.get("n_items", 0) if health else 0
    n_users = health.get("n_users", 0) if health else 0
    is_online = health and health.get("status") == "healthy"
    
    status_class = "online" if is_online else "offline"
    status_text = "En ligne" if is_online else "Hors ligne"
    status_icon = "fa-circle" if is_online else "fa-exclamation-triangle"
    
    st.markdown(f'''
    <div class="app-header">
        <div class="header-content">
            <h1 class="header-title">
                <i class="fas fa-film"></i>
                Recommandation Films
            </h1>
            <p class="header-tagline">Decouvrez des films adaptes a vos gouts</p>
        </div>
        <div class="header-meta">
            <span class="status-badge {status_class}">
                <i class="fas {status_icon}"></i>
                {status_text}
            </span>
            <div class="header-stats">
                <span class="header-stat">
                    <i class="fas fa-database"></i>
                    {n_films:,} films
                </span>
                <span class="header-stat">
                    <i class="fas fa-users"></i>
                    {n_users:,} profils
                </span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def welcome_section() -> None:
    """Affiche un guide rapide pour les nouveaux utilisateurs."""
    with st.expander("Comment utiliser cette application ?", expanded=False):
        st.markdown("""
        **Bienvenue !** Cette application vous aide a decouvrir des films adaptes a vos gouts.
        
        Vous explorez des **profils de demonstration**. Chaque profil represente un utilisateur avec des gouts differents.
        
        **Navigation :**
        
        - **Recommandations** : Obtenez des suggestions personnalisees selon un profil utilisateur
        - **Decouvrir** : Trouvez des films similaires a ceux que vous aimez
        - **Historique** : Consultez les films qu'un utilisateur a deja notes
        - **Statistiques** : Voyez les chiffres cles du systeme
        
        *Selectionnez un onglet dans le menu pour commencer.*
        """)


def tab_recommendations() -> None:
    """Onglet Recommandations avec UX amelioree."""
    st.markdown('''
    <div class="section-header">
        <i class="fas fa-star"></i>
        <h3>Recommandations Personnalisees</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-card">
        <i class="fas fa-lightbulb"></i>
        <div class="content">
            <p>Notre algorithme analyse les preferences du profil selectionne pour suggerer des films pertinents.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    users = api_users()
    if not users:
        st.warning("Impossible de charger les profils utilisateurs.")
        st.caption("Verifiez que le service API est disponible et reessayez.")
        return
    
    st.markdown("#### Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.selectbox(
            "Selectionnez un profil",
            users,
            format_func=lambda x: f"Utilisateur #{x}",
            help="Chaque profil represente un utilisateur avec un historique de notes unique",
            key="rec_user"
        )
    
    with col2:
        k = st.select_slider(
            "Nombre de recommandations",
            options=[5, 10, 15, 20, 30],
            value=10,
            help="Plus le nombre est eleve, plus vous aurez de suggestions",
            key="rec_k"
        )
    
    # Progressive disclosure for advanced options
    with st.expander("Options avancees", expanded=False):
        exclude = st.checkbox(
            "Exclure les films deja vus",
            value=True,
            help="Cochez pour ne voir que des films pas encore notes",
            key="rec_exclude"
        )
        st.caption("Par defaut, nous excluons les films de l'historique pour privilegier la decouverte.")
    
    # Get value from session state
    exclude = st.session_state.get("rec_exclude", True)
    
    st.markdown("")
    
    if st.button("Obtenir les recommandations", type="primary", use_container_width=True, key="rec_btn"):
        progress_placeholder = st.empty()
        progress_placeholder.info("Analyse du profil en cours...")
        
        with st.spinner(""):
            data = api_recommend(user_id, k, exclude)
        
        progress_placeholder.empty()
        
        if data and data.get("recommendations"):
            recs = data["recommendations"]
            st.success(f"{len(recs)} films recommandes")
            
            for i in range(0, len(recs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(recs):
                        rec = recs[i + j]
                        with col:
                            with st.container():
                                title = rec.get("title") or f"Film #{rec['item_id']}"
                                score = rec["score"]
                                score_label = get_score_label(score)
                                
                                st.markdown(f"**{title}**")
                                if rec.get("genres"):
                                    st.caption(rec.get("genres"))
                                st.progress(min(score, 1.0))
                                st.caption(f"{score_label} ({score:.0%})")
        else:
            st.info("Aucune recommandation disponible pour ce profil.")
            st.caption("Essayez de decocher l'option 'Exclure les films deja vus' dans les options avancees.")


def tab_discover() -> None:
    """Onglet Decouvrir (films similaires) avec UX amelioree."""
    st.markdown('''
    <div class="section-header">
        <i class="fas fa-compass"></i>
        <h3>Explorer des Films Similaires</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-card">
        <i class="fas fa-bullseye"></i>
        <div class="content">
            <p>Selectionnez un film que vous aimez, et nous trouverons d'autres films avec des caracteristiques similaires.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    items = api_items_with_metadata()
    if not items:
        st.warning("Le catalogue de films n'est pas disponible.")
        st.caption("Le service semble temporairement indisponible. Reessayez dans quelques instants.")
        return
    
    # Create a mapping for display
    item_options = {item["item_id"]: item["title"] for item in items}
    
    st.markdown("#### Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        item_id = st.selectbox(
            "Choisissez un film de reference",
            options=list(item_options.keys()),
            format_func=lambda x: item_options.get(x, f"Film #{x}"),
            help="Selectionnez un film que vous appreciez pour trouver des films similaires",
            key="sim_item"
        )
    
    with col2:
        k = st.select_slider(
            "Nombre de resultats",
            options=[5, 10, 15, 20],
            value=10,
            help="Nombre de films similaires a afficher",
            key="sim_k"
        )
    
    st.markdown("")
    
    if st.button("Trouver des films similaires", type="primary", use_container_width=True, key="sim_btn"):
        progress_placeholder = st.empty()
        progress_placeholder.info("Recherche de films similaires...")
        
        with st.spinner(""):
            data = api_similar(item_id, k)
        
        progress_placeholder.empty()
        
        if data and data.get("similar_items"):
            sims = data["similar_items"]
            st.success(f"{len(sims)} films similaires trouves")
            
            df = pd.DataFrame([
                {
                    "Titre": s.get("title", f"Film #{s['item_id']}"),
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
            st.info("Aucun film similaire trouve pour cette selection.")
            st.caption("Essayez de selectionner un autre film de reference.")


def tab_history() -> None:
    """Onglet Historique avec UX amelioree."""
    st.markdown('''
    <div class="section-header">
        <i class="fas fa-history"></i>
        <h3>Historique des Notes</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-card">
        <i class="fas fa-folder-open"></i>
        <div class="content">
            <p>Consultez les films qu'un utilisateur a deja notes. Ces donnees alimentent nos recommandations.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    users = api_users()
    if not users:
        st.warning("Les profils utilisateurs ne sont pas disponibles.")
        return
    
    user_id = st.selectbox(
        "Selectionnez un profil",
        users,
        format_func=lambda x: f"Utilisateur #{x}",
        help="Choisissez le profil dont vous souhaitez voir l'historique",
        key="hist_user"
    )
    
    st.markdown("")
    
    if st.button("Voir l'historique", type="primary", use_container_width=True, key="hist_btn"):
        progress_placeholder = st.empty()
        progress_placeholder.info("Chargement de l'historique...")
        
        with st.spinner(""):
            data = api_history(user_id)
        
        progress_placeholder.empty()
        
        if data and data.get("history"):
            hist = data["history"]
            st.success(f"{data['count']} films dans l'historique")
            
            df = pd.DataFrame([
                {
                    "Titre": h.get("title", f"Film #{h['item_id']}"),
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
                        format="%.1f / 5.0",
                    ),
                },
            )
        else:
            st.info("Aucun film note pour ce profil.")
            st.caption("Cet utilisateur n'a pas encore evalue de films.")


def tab_stats() -> None:
    """Onglet Statistiques avec UX amelioree."""
    st.markdown('''
    <div class="section-header">
        <i class="fas fa-chart-line"></i>
        <h3>Statistiques du Systeme</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-card">
        <i class="fas fa-cogs"></i>
        <div class="content">
            <p>Vue d'ensemble du moteur de recommandation et de ses donnees.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    health = api_health()
    
    if health:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Utilisateurs",
                value=health.get("n_users", "N/A"),
                help="Nombre total de profils utilisateurs dans le systeme"
            )
        
        with col2:
            st.metric(
                label="Films",
                value=health.get("n_items", "N/A"),
                help="Nombre total de films disponibles dans le catalogue"
            )
        
        with col3:
            status = "En ligne" if health.get("status") == "healthy" else "Degrade"
            st.metric(
                label="Statut API",
                value=status,
                help="Etat actuel du service de recommandation"
            )
        
        st.markdown("---")
        
        with st.expander("A propos du systeme", expanded=False):
            st.markdown("""
            **Moteur de Recommandation**
            
            Ce systeme utilise des algorithmes de filtrage collaboratif pour generer 
            des recommandations personnalisees. Il analyse les historiques de notation 
            pour identifier des patterns et suggerer des films pertinents.
            
            **Caracteristiques :**
            
            - **Algorithme** : Filtrage collaboratif (Matrix Factorization)
            - **Mise a jour** : Modele re-entraine periodiquement
            - **Latence** : Recommandations en temps reel (< 1 seconde)
            """)
    else:
        st.error("Impossible de recuperer les statistiques du systeme.")
        st.caption("Le service API semble indisponible. Verifiez la connexion et reessayez.")
        
        if st.button("Reessayer", key="stats_retry"):
            st.cache_data.clear()
            st.rerun()


# =============================================================================
# Application principale
# =============================================================================

def main() -> None:
    """Point d'entree."""
    load_css()
    
    health = api_health()
    if not health or health.get("status") != "healthy":
        st.error("Le service de recommandation n'est pas disponible.")
        
        with st.expander("Que faire ?", expanded=True):
            st.markdown(f"""
            **Causes possibles :**
            - Le serveur API n'est pas demarre
            - Probleme de connexion reseau
            - Le service est en cours de redemarrage
            
            **Actions suggerees :**
            1. Verifiez que l'API est accessible sur `{API_URL}`
            2. Attendez quelques instants et rechargez la page
            3. Contactez l'administrateur si le probleme persiste
            """)
        
        if st.button("Reessayer maintenant", type="primary"):
            st.cache_data.clear()
            st.rerun()
        return
    
    header()
    welcome_section()
    
    st.markdown("---")
    
    # Initialize tab state
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Recommandations"
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown('''
        <div class="sidebar-brand">
            <i class="fas fa-film"></i>
            <span>Recommandation Films</span>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        st.markdown('<p class="sidebar-nav-title"><i class="fas fa-bars"></i> Navigation</p>', unsafe_allow_html=True)
        
        menu_items = {
            "Recommandations": "fa-star",
            "Decouvrir": "fa-compass", 
            "Historique": "fa-history",
            "Statistiques": "fa-chart-line"
        }
        
        for item, icon in menu_items.items():
            is_active = st.session_state.current_tab == item
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                f"  {item}",
                key=f"nav_{item}",
                type=btn_type,
                use_container_width=True
            ):
                st.session_state.current_tab = item
                st.rerun()
        
        st.markdown("---")
        
        # Status Indicators
        health = api_health()
        if health and health.get("status") == "healthy":
            n_users = health.get("n_users", 0)
            n_items = health.get("n_items", 0)
            st.markdown(f'''
            <div class="sidebar-status success">
                <div class="status-header">
                    <i class="fas fa-check-circle"></i>
                    <span>Systeme Operationnel</span>
                </div>
                <div class="status-details">
                    <div class="status-item">
                        <i class="fas fa-database"></i>
                        <span>{n_items:,} films</span>
                    </div>
                    <div class="status-item">
                        <i class="fas fa-users"></i>
                        <span>{n_users:,} utilisateurs</span>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="sidebar-status error">
                <div class="status-header">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Service Indisponible</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer Links
        st.markdown('''
        <div class="sidebar-links">
            <p class="sidebar-nav-title"><i class="fas fa-external-link-alt"></i> Liens</p>
            <a href="https://github.com/Souley225/mlops-recommender-system" target="_blank">
                <i class="fab fa-github"></i> Code Source
            </a>
            <a href="https://github.com/Souley225" target="_blank">
                <i class="fab fa-github"></i> GitHub
            </a>
            <a href="https://www.linkedin.com/in/souleymanes-sall/" target="_blank">
                <i class="fab fa-linkedin"></i> LinkedIn
            </a>
        </div>
        ''', unsafe_allow_html=True)
    
    # Display active tab content
    if st.session_state.current_tab == "Recommandations":
        tab_recommendations()
    elif st.session_state.current_tab == "Decouvrir":
        tab_discover()
    elif st.session_state.current_tab == "Historique":
        tab_history()
    elif st.session_state.current_tab == "Statistiques":
        tab_stats()
    
    # Footer
    st.markdown('''
    <div class="footer">
        <div class="footer-content">
            <div class="footer-brand">
                <h4><i class="fas fa-film"></i> Systeme de Recommandation de Films</h4>
                <p>Moteur de recommandation intelligent propulse par le Machine Learning</p>
            </div>
            <div class="footer-tech">
                <p class="footer-label"><i class="fas fa-code"></i> Stack Technique</p>
                <div class="footer-badges">
                    <span class="footer-badge"><i class="fab fa-python"></i> Python</span>
                    <span class="footer-badge"><i class="fas fa-bolt"></i> FastAPI</span>
                    <span class="footer-badge"><i class="fas fa-chart-bar"></i> Streamlit</span>
                    <span class="footer-badge"><i class="fas fa-brain"></i> Scikit-learn</span>
                </div>
            </div>
            <div class="social-links">
                <a href="https://github.com/Souley225" target="_blank" title="GitHub"><i class="fab fa-github"></i></a>
                <a href="https://www.linkedin.com/in/souleymanes-sall/" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
                <a href="mailto:sallsouleymane2207@gmail.com" title="Email"><i class="fas fa-envelope"></i></a>
            </div>
            <a href="https://github.com/Souley225/mlops-recommender-system" target="_blank" class="source-code-link">
                <i class="fas fa-code-branch"></i> Code Source
            </a>
            <p class="footer-license"><i class="fas fa-balance-scale"></i> MIT License - 2026</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
