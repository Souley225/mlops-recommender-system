# Système de Recommandation de Films

> Découvrez vos prochains films préférés grâce à l'intelligence artificielle

[![Demo](https://img.shields.io/badge/Demo-Live-success?style=flat-square)](https://mlops-recommender-ui.onrender.com)
[![API](https://img.shields.io/badge/API-Docs-blue?style=flat-square)](https://mlops-recommender-system-1.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)

---

## Présentation

Ce projet est un **moteur de recommandation de films** intelligent qui analyse les goûts des utilisateurs pour suggérer des films personnalisés.

Comparable aux systèmes utilisés par Netflix ou Amazon : le système apprend des préférences utilisateurs et recommande des contenus similaires.

### Cas d'usage

| Scénario | Fonctionnement |
|----------|----------------|
| Utilisateur a aimé "Inception" | Recommande des thrillers sci-fi similaires |
| Nouvel utilisateur | Propose les films les plus populaires |
| Exploration | "Les gens qui ont aimé ce film ont aussi aimé..." |

---

## Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| Recommandations personnalisées | Suggestions adaptées à chaque utilisateur |
| Films similaires | Trouver des films qui ressemblent aux favoris |
| Interface intuitive | Application web simple à utiliser |
| API REST | Intégration facile dans d'autres applications |

---

## Démonstration

### Interface utilisateur

L'application Streamlit permet de :
- Sélectionner un profil utilisateur
- Obtenir des recommandations personnalisées
- Explorer des films similaires
- Consulter l'historique des interactions

**Accès :** [mlops-recommender-ui.onrender.com](https://mlops-recommender-ui.onrender.com)

### API REST

Documentation interactive Swagger disponible.

**Accès :** [mlops-recommender-system-1.onrender.com/docs](https://mlops-recommender-system-1.onrender.com/docs)

---

## Stack technique

| Domaine | Technologies |
|---------|-------------|
| Data Science | Python, Pandas, Scikit-learn |
| Machine Learning | Filtrage collaboratif, ALS |
| API | FastAPI, Pydantic |
| Frontend | Streamlit |
| DevOps | Docker, CI/CD |
| Cloud | Render |
| MLOps | MLflow, DVC |

---

## Données

Jeu de données **MovieLens** :
- ~100 000 évaluations
- ~600 utilisateurs
- ~10 000 films

---

## Fonctionnement

### Pipeline

1. **Collecte** : Récupération des notes données par les utilisateurs
2. **Analyse** : Détection de patterns comportementaux
3. **Prédiction** : Estimation des films non vus que l'utilisateur aimerait
4. **Recommandation** : Affichage des meilleures prédictions

### Filtrage Collaboratif

Le système analyse les **comportements utilisateurs similaires** plutôt que le contenu des films. Si deux utilisateurs ont aimé les mêmes films, leurs autres préférences sont probablement similaires.

---

## Installation locale

```bash
# Cloner le projet
git clone https://github.com/Souley225/mlops-recommender-system.git
cd mlops-recommender-system

# Installation
pip install poetry
poetry install

# Lancer l'API
poetry run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Lancer l'interface (autre terminal)
poetry run streamlit run src/ui/app.py
```

---

## Structure

```
mlops-recommender-system/
├── src/
│   ├── models/      # Algorithmes de recommandation
│   ├── serving/     # API REST
│   └── ui/          # Interface Streamlit
├── data/            # Données MovieLens
├── models/          # Modèles entraînés
└── docker/          # Configuration Docker
```

---

## Contact

**Souleymane SALL** - Data Scientist / ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-Souley225-181717?style=flat-square&logo=github)](https://github.com/Souley225)

---

## Licence

MIT
