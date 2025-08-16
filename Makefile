.PHONY: install test lint format clean pipeline smoke-test docker-build docker-run

# Development
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest -q

test-unit:
	pytest tests/unit/ -q

test-integration:
	pytest tests/integration/ -q

test-verbose:
	pytest -v

test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/unit/ tests/integration/ --max-line-length=88 --ignore=E203,W503,E402

lint-strict:
	flake8 src/ tests/unit/ tests/integration/ --max-line-length=88 --ignore=E203,W503 --count --select=E9,F63,F7,F82 --show-source --statistics

format:
	black src/ tests/unit/ tests/integration/ --line-length=88

format-check:
	black --check --line-length=88 src/ tests/unit/ tests/integration/

sort-imports:
	isort src/ tests/unit/ tests/integration/ --profile black

sort-imports-check:
	isort --check-only --profile black src/ tests/unit/ tests/integration/

type-check:
	mypy src/ --ignore-missing-imports

pre-commit-all:
	pre-commit run --all-files

clean:
	powershell -Command "if (Test-Path 'mlruns') { Remove-Item -Recurse -Force 'mlruns' }"
	powershell -Command "if (Test-Path 'artifacts') { Remove-Item -Recurse -Force 'artifacts' }"
	powershell -Command "if (Test-Path 'data\\components') { Remove-Item -Recurse -Force 'data\\components' }"
	powershell -Command "if (Test-Path 'data\\monitoring') { Remove-Item -Recurse -Force 'data\\monitoring' }"
	powershell -Command "if (Test-Path 'data\\weights') { Remove-Item -Recurse -Force 'data\\weights' }"
	powershell -Command "if (Test-Path 'data') { Remove-Item -Force 'data' }"
	powershell -Command "Get-ChildItem -Recurse -Filter '*.pyc' | Remove-Item -Force"
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force"
	echo "Cleanup completed"

# Pipeline
pipeline:
	python -m src.run_pipeline --interval 1h --days 30 --max-symbols 10

smoke-test:
	python -m src.run_pipeline --interval 1h --days 7 --max-symbols 5
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &
	powershell -Command "Start-Sleep -Seconds 5"
	curl -f http://localhost:8000/health
	curl -f "http://localhost:8000/predict/component?symbol=AAPL"
	echo "Smoke test completed successfully. API server is still running on http://localhost:8000"

# Docker (optional)
docker-build:
	docker build -t market-master-api .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# Registry management
promote-staging:
	python -m src.registry promote-staging

rollback-production:
	python -m src.registry rollback-production

# Help
help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install        - Install all dependencies (including dev tools)"
	@echo "  install-dev    - Install dependencies and set up pre-commit hooks"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-verbose   - Run tests with verbose output"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  lint           - Run linting"
	@echo "  lint-strict    - Run strict linting"
	@echo "  format         - Format code with black"
	@echo "  format-check   - Check code formatting"
	@echo "  sort-imports   - Sort imports with isort"
	@echo "  sort-imports-check - Check import sorting"
	@echo "  type-check     - Run type checking with mypy"
	@echo "  pre-commit-all - Run all pre-commit hooks"
	@echo "  clean          - Clean artifacts and cache"
	@echo ""
	@echo "Pipeline:"
	@echo "  pipeline       - Run full pipeline"
	@echo "  smoke-test     - Run smoke test"
	@echo "  mlflow-ui      - Start MLflow UI"
	@echo "  promote-staging - Promote model to staging"
	@echo "  rollback-production - Rollback production model"
