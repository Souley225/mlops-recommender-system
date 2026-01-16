# -*- coding: utf-8 -*-
"""
Test du chargement du modèle de recommandation.

Ce module teste que le Recommender peut charger correctement le modèle
et accéder aux données utilisateurs.
"""

import pytest


class TestRecommenderLoading:
    """Tests pour le chargement du Recommender."""

    def test_recommender_loads_successfully(self):
        """Test que le Recommender se charge sans erreur."""
        from src.models.recommend import Recommender

        recommender = Recommender()
        recommender.load()
        
        users = recommender.get_all_users()
        assert len(users) > 0, "Aucun utilisateur chargé"
        
    def test_recommender_can_recommend(self):
        """Test que le Recommender peut générer des recommandations."""
        from src.models.recommend import Recommender

        recommender = Recommender()
        recommender.load()
        
        users = recommender.get_all_users()
        if len(users) > 0:
            first_user = users[0]
            recs = recommender.recommend(first_user, n=5)
            assert len(recs) > 0, "Aucune recommandation générée"
