# Guide d'Installation et de Configuration

Ce guide explique comment installer, configurer et deployer le systeme de recommandation MLOps.

## Prerequis

- Python 3.11+
- Poetry 1.7+
- Git
- Docker (optionnel)
- DVC (optionnel, installe via Poetry)

## Installation Locale

### 1. Cloner le Repository

```bash
git clone https://github.com/your-org/mlops-recommender-system.git
cd mlops-recommender-system
```

### 2. Installer les Dependances

```bash
# Installer Poetry si necessaire
pip install poetry

# Installer les dependances du projet
poetry install

# Activer l'environnement virtuel
poetry shell
```

### 3. Configuration Pre-commit

```bash
poetry run pre-commit install
```

### 4. Initialiser DVC

```bash
dvc init
```

## Entrainement du Modele

### Option 1: Pipeline DVC Complet

```bash
# Executer le pipeline complet
dvc repro

# Ou etape par etape
dvc repro download_data
dvc repro make_dataset
dvc repro split_dataset
dvc repro build_features
dvc repro train
dvc repro evaluate
dvc repro register
```

### Option 2: Makefile

```bash
make train
```

### Option 3: Execution Directe

```bash
# Telecharger les donnees
python -m src.data.download_data

# Creer le dataset
python -m src.data.make_dataset

# Diviser le dataset
python -m src.data.split_dataset

# Construire les features
python -m src.features.build_features

# Entrainer le modele
python -m src.models.train

# Evaluer le modele
python -m src.models.evaluate

# Enregistrer le modele
python -m src.models.register
```

## Lancer l'API

### Mode Developpement

```bash
# Avec Makefile
make serve

# Ou directement
poetry run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

L'API sera disponible sur http://localhost:8000

Documentation Swagger: http://localhost:8000/docs

### Mode Production

```bash
poetry run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Lancer l'Interface Streamlit

```bash
# Avec Makefile
make ui

# Ou directement
poetry run streamlit run src/ui/app.py
```

L'UI sera disponible sur http://localhost:8501

## MLflow UI

```bash
# Lancer le serveur MLflow
make mlflow

# Ou directement
poetry run mlflow ui --host 0.0.0.0 --port 5000
```

MLflow UI sera disponible sur http://localhost:5000

## Docker

### Construire les Images

```bash
# Construire toutes les images
docker compose build

# Ou individuellement
docker build -f docker/Dockerfile.api -t mlops-recommender-api .
docker build -f docker/Dockerfile.ui -t mlops-recommender-ui .
docker build -f docker/Dockerfile.train -t mlops-recommender-train .
```

### Lancer les Services

```bash
# Lancer API, UI et MLflow
docker compose up -d

# Voir les logs
docker compose logs -f

# Arreter les services
docker compose down
```

### Lancer l'Entrainement

```bash
docker compose --profile training up train
```

## Deploiement Render

### Configuration API

1. Creer un nouveau **Web Service** sur Render
2. Connecter le repository GitHub
3. Configurer:
   - **Name**: `mlops-recommender-api`
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install poetry && poetry config virtualenvs.create false && poetry install --only main
     ```
   - **Start Command**:
     ```bash
     uvicorn src.serving.api:app --host 0.0.0.0 --port $PORT
     ```
4. Variables d'environnement:
   - `MLFLOW_TRACKING_URI`: `mlruns` (ou URI externe)
   - `LOG_LEVEL`: `INFO`
   - `LOG_FORMAT`: `json`

### Configuration UI (Optionnel)

1. Creer un nouveau **Web Service**
2. Configurer:
   - **Name**: `mlops-recommender-ui`
   - **Build Command**:
     ```bash
     pip install poetry && poetry config virtualenvs.create false && poetry install --only main
     ```
   - **Start Command**:
     ```bash
     streamlit run src/ui/app.py --server.port $PORT --server.address 0.0.0.0
     ```
3. Variables d'environnement:
   - `API_URL`: URL de l'API deployee

### Stockage des Modeles

Pour le deploiement, les modeles doivent etre accessibles. Options:

1. **Inclure dans l'image Docker** (simple, moins flexible)
2. **Stockage S3/GCS** avec chargement au demarrage
3. **Volume persistant** sur Render

## Configuration

### Fichier params.yaml

```yaml
# Donnees
data:
  dataset_name: ml-latest-small
  raw_dir: data/raw
  processed_dir: data/processed

# Modele
model:
  type: als
  embedding_dim: 64
  regularization: 0.01
  iterations: 15
  alpha: 40

# Entrainement
train:
  random_seed: 42
  log_to_mlflow: true
  experiment_name: mlops-recommender

# HPO
optuna:
  enabled: true
  n_trials: 20
  metric: ndcg_at_10
```

### Variables d'Environnement

Copier `.env.example` vers `.env`:

```bash
cp .env.example .env
```

Variables principales:

| Variable | Description | Defaut |
|----------|-------------|--------|
| `MLFLOW_TRACKING_URI` | URI du serveur MLflow | `mlruns` |
| `API_HOST` | Host de l'API | `0.0.0.0` |
| `API_PORT` | Port de l'API | `8000` |
| `LOG_LEVEL` | Niveau de log | `INFO` |

## Tests

```bash
# Tous les tests
make test

# Tests rapides (sans couverture)
make test-fast

# Avec pytest directement
poetry run pytest tests/ -v
```

## Linting

```bash
# Verifier le code
make lint

# Formater le code
make format
```

## Resolution des Problemes

### Erreur: Module not found

```bash
# Reinstaller les dependances
poetry install

# Verifier l'environnement
poetry env info
```

### Erreur: MLflow connection refused

```bash
# Lancer le serveur MLflow
make mlflow

# Ou utiliser le stockage local
export MLFLOW_TRACKING_URI=mlruns
```

### Erreur: API model not loaded

Verifier que le modele existe:

```bash
ls models/model.joblib

# Si absent, entrainer le modele
make train
```

### Docker: Permission denied

```bash
# Linux: ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER
# Redemarrer la session
```

## Support

Pour toute question ou probleme, ouvrir une issue sur GitHub.
