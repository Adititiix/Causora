# Use the project's virtualenv automatically if `make setup` created one,
# so you never have to remember to activate it.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help setup up down reset-db logs psql data test test-fast lint fmt api

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## Create .venv, install dev dependencies, copy .env.example to .env
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt
	.venv/bin/pip install -e . --no-deps
	@test -f .env || cp .env.example .env
	@echo "Setup done. Edit .env if you want, then run: make up"

up: ## Start Postgres + API in Docker (needs .env)
	docker compose up -d --build

down: ## Stop containers (keeps your data)
	docker compose down

reset-db: ## Wipe the database volume and re-run db/init/*.sql
	docker compose down -v
	docker compose up -d db

logs: ## Tail API logs
	docker compose logs -f api

psql: ## Open a psql shell inside the db container
	docker compose exec db sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

data: ## Regenerate the synthetic dataset and seed eval questions
	$(PY) scripts/generate_synthetic_data.py --out data --eval-out eval/questions.seed.jsonl

test: ## Run all tests (database tests skip if Postgres is unreachable)
	$(PY) -m pytest -v

test-fast: ## Run only the tests that need no database
	$(PY) -m pytest -v -m "not db"

lint: ## Check code style with ruff
	$(PY) -m ruff check .

fmt: ## Auto-fix style problems
	$(PY) -m ruff check --fix .

api: ## Run the API on your machine (not in Docker) with auto-reload
	$(PY) -m uvicorn incident_copilot.api.main:app --reload --port 8000 --app-dir src
