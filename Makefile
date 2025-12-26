# Makefile pour le systeme de recommandation MLOps
# Commandes utilitaires pour le developpement et le deploiement

.PHONY: install setup lint format test train serve ui docker-build docker-up clean help

# Variables
PYTHON := poetry run python
PYTEST := poetry run pytest
UVICORN := poetry run uvicorn
STREAMLIT := poetry run streamlit

help:  ## Afficher l'aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installer les dependances avec Poetry
	poetry install

setup: install  ## Configuration complete du projet
	poetry run pre-commit install
	dvc init --no-scm || true
	mkdir -p data/raw data/interim data/processed models

lint:  ## Verifier le code avec ruff, black et mypy
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run mypy src/

format:  ## Formater le code avec black et isort
	poetry run black src/ tests/
	poetry run isort src/ tests/
	poetry run ruff check --fix src/ tests/

test:  ## Executer les tests avec pytest
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-fast:  ## Executer les tests sans couverture
	$(PYTEST) tests/ -v -x

train:  ## Entrainer le modele via DVC
	dvc repro

train-force:  ## Forcer le re-entrainement complet
	dvc repro --force

serve:  ## Lancer l'API FastAPI
	$(UVICORN) src.serving.api:app --host 0.0.0.0 --port 8000 --reload

serve-prod:  ## Lancer l'API en mode production
	$(UVICORN) src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

ui:  ## Lancer l'interface Streamlit
	$(STREAMLIT) run src/ui/app.py --server.port 8501 --server.address 0.0.0.0

mlflow:  ## Lancer le serveur MLflow
	poetry run mlflow ui --host 0.0.0.0 --port 5000

docker-build:  ## Construire les images Docker
	docker compose build

docker-up:  ## Lancer les services Docker
	docker compose up -d

docker-down:  ## Arreter les services Docker
	docker compose down

docker-logs:  ## Afficher les logs Docker
	docker compose logs -f

clean:  ## Nettoyer les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

clean-data:  ## Nettoyer les donnees et modeles
	rm -rf data/raw/* data/interim/* data/processed/* models/*

reset:  ## Reinitialiser completement le projet
	$(MAKE) clean
	$(MAKE) clean-data
	dvc repro
