# MLOps Recommender System

Systeme de recommandation de niveau production avec MovieLens, suivant les meilleures pratiques MLOps.

## Architecture

```
                                    +------------------+
                                    |   MLflow UI      |
                                    |   (Tracking)     |
                                    +--------+---------+
                                             |
+------------------+     +------------------+|     +------------------+
|   DVC Pipeline   |---->|   Training       |+---->|   Model          |
|   (Orchestration)|     |   (ALS/SVD)      |      |   Registry       |
+------------------+     +------------------+      +--------+---------+
                                                           |
                         +------------------+              |
                         |   FastAPI        |<-------------+
                         |   (Serving)      |
                         +--------+---------+
                                  |
                         +--------+---------+
                         |   Streamlit      |
                         |   (UI)           |
                         +------------------+
```

## Fonctionnalites

- **Modeles**: Popularite baseline, ALS (Alternating Least Squares)
- **Pipeline**: DVC pour la reproductibilite complete
- **Tracking**: MLflow pour les experiences et le registre de modeles
- **HPO**: Optuna pour l'optimisation des hyperparametres
- **API**: FastAPI avec endpoints REST
- **UI**: Streamlit pour l'exploration interactive
- **CI/CD**: GitHub Actions
- **Deploiement**: Docker Compose + Render

## Structure du Projet

```
mlops-recommender-system/
  README.md                 # Ce fichier
  SETUP.md                  # Guide d'installation
  pyproject.toml            # Configuration Python/Poetry
  params.yaml               # Parametres du pipeline
  dvc.yaml                  # Definition du pipeline DVC
  compose.yaml              # Docker Compose
  Makefile                  # Commandes utilitaires
  
  configs/                  # Configuration Hydra
    data.yaml
    model.yaml
    train.yaml
    eval.yaml
    serving.yaml
  
  src/
    data/                   # Pipeline de donnees
      download_data.py
      make_dataset.py
      split_dataset.py
    
    features/               # Ingenierie des features
      build_features.py
    
    models/                 # Entrainement et evaluation
      train.py
      evaluate.py
      recommend.py
      register.py
    
    serving/                # API REST
      api.py
    
    ui/                     # Interface Streamlit
      app.py
    
    utils/                  # Utilitaires
      paths.py
      io.py
      logging.py
      mlflow_utils.py
      metrics.py
      reproducibility.py
  
  tests/                    # Tests unitaires
  docker/                   # Dockerfiles
  .github/workflows/        # CI/CD
```

## Algorithmes

### Popularite (Baseline)

Recommande les items les plus populaires. Sert de reference pour evaluer les modeles plus sophistiques.

### ALS (Alternating Least Squares)

Factorisation matricielle optimisee pour le feedback implicite. Decompose la matrice user-item en facteurs latents.

**Hyperparametres principaux**:
- `embedding_dim`: Dimension des facteurs latents
- `regularization`: Regularisation L2
- `iterations`: Nombre d'iterations
- `alpha`: Parametre de confiance

## Metriques d'Evaluation

| Metrique | Description |
|----------|-------------|
| Precision@K | Proportion d'items pertinents parmi les K recommandes |
| Recall@K | Proportion d'items pertinents retrouves |
| NDCG@K | Qualite du ranking avec penalite de position |
| MAP@K | Precision moyenne aux positions de rappel |
| MRR | Rang reciproque moyen du premier item pertinent |

## API Endpoints

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Verification de l'etat de sante |
| POST | `/recommend` | Recommandations pour un utilisateur |
| POST | `/similar-items` | Items similaires a un item |
| GET | `/users` | Liste des utilisateurs |
| GET | `/items` | Liste des items |

### Exemple de requete

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "k": 10}'
```

## Commandes Rapides

```bash
# Installation
make install

# Entrainement complet
make train

# Lancer l'API
make serve

# Lancer l'UI
make ui

# Tests
make test

# Linting
make lint

# Docker
make docker-build
make docker-up
```

## Configuration

Les parametres sont definis dans `params.yaml`:

```yaml
model:
  type: als          # popularity, als
  embedding_dim: 64
  regularization: 0.01
  iterations: 15

optuna:
  enabled: true
  n_trials: 20
  metric: ndcg_at_10
```

## Deploiement Render

1. Pousser le code sur GitHub
2. Creer un nouveau Web Service sur Render
3. Configurer:
   - Build Command: `pip install poetry && poetry install`
   - Start Command: `poetry run uvicorn src.serving.api:app --host 0.0.0.0 --port $PORT`
4. Ajouter les variables d'environnement

Voir [SETUP.md](SETUP.md) pour les instructions detaillees.

## Licence

MIT
