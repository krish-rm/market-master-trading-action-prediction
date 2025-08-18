.PHONY: install test lint format clean pipeline smoke-test docker-build docker-run

# Quick Reference:
# Option A (Local): 
#   - Simple: make pipeline (uses src.run_pipeline.py)
#   - Enhanced: make prefect-flow (uses flows/enhanced_orchestration.py)
# Option B (Docker): 
#   - Complete: make docker-setup (uses src.run_pipeline.py in containers)

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

# Option A: Local Pipeline (Simple)
pipeline:
	python -m src.run_pipeline --interval 1h --days 30 --max-symbols 10

smoke-test:
	python -m src.run_pipeline --interval 1h --days 7 --max-symbols 5
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &
	powershell -Command "Start-Sleep -Seconds 5"
	curl -f http://localhost:8000/health
	curl -f "http://localhost:8000/predict/component?symbol=AAPL"
	echo "Smoke test completed successfully. API server is still running on http://localhost:8000"

# Option A: Local Pipeline (Enhanced Prefect Orchestration)
prefect-start:
	prefect server start --host 127.0.0.1 --port 4200

prefect-setup:
	prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

prefect-worker:
	prefect worker start -p process -q default

prefect-flow:
	python flows/enhanced_orchestration.py

prefect-deploy:
	prefect deployment build flows/enhanced_orchestration.py:enhanced_index_flow -n "enhanced-index-signal-hourly"

prefect-deploy-apply:
	prefect deployment apply enhanced_index_flow-deployment.yaml

prefect-deploy-all:
	prefect deployment build flows/enhanced_orchestration.py:enhanced_index_flow -n "enhanced-index-signal-hourly" --cron "0 * * * *" --timezone "UTC" --params '{"interval": "1h", "days": 7, "max_symbols": 5}'
	prefect deployment build flows/enhanced_orchestration.py:enhanced_index_flow -n "enhanced-index-signal-daily" --cron "0 0 * * *" --timezone "UTC" --params '{"interval": "1h", "days": 30, "max_symbols": 10}'
	prefect deployment build flows/enhanced_orchestration.py:enhanced_index_flow -n "enhanced-index-signal-weekly" --cron "0 0 * * 0" --timezone "UTC" --params '{"interval": "1h", "days": 60, "max_symbols": null}'

# Model Serving (MLflow)
model-serving:
	python -m src.model_serving

model-serving-test:
	uvicorn src.model_serving:app --host 0.0.0.0 --port 8001 --reload &
	powershell -Command "Start-Sleep -Seconds 10"
	curl -f http://localhost:8001/health
	curl -f "http://localhost:8001/model-info"
	echo "Model serving test completed. Server running on http://localhost:8001"

model-serving-simple:
	echo "Model serving test completed. Use 'make model-serving-test' for full API testing."

# Streamlit Dashboard
streamlit-dashboard:
	streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address localhost

# Option B: Docker Pipeline (Optimized Reproducible Setup)
docker-build:
	docker compose build

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart

docker-clean:
	docker compose down -v
	docker system prune -a --volumes -f

docker-setup:
	@echo "Setting up optimized Docker environment..."
	@echo "1. Building Docker images..."
	make docker-build
	@echo "2. Starting services..."
	make docker-run
	@echo "3. Running pipeline..."
	make docker-pipeline
	@echo "Docker setup complete! Access services at:"
	@echo "- API: http://localhost:8000/docs"
	@echo "- MLflow UI: http://localhost:5000"
	@echo "- Dashboard: http://localhost:8501"

docker-pipeline:
	@echo "Running ML pipeline in Docker..."
	@echo "This will train the model and register it in MLflow..."
	docker compose exec -T api python -m src.run_pipeline
	@echo "Pipeline completed. Model should now be available in MLflow registry."

docker-smoke-test:
	@echo "Testing API endpoints..."
	curl -f http://localhost:8000/health
	curl -f "http://localhost:8000/predict/component?symbol=AAPL"
	@echo "Smoke test completed successfully!"

docker-health-check:
	@echo "Checking service health..."
	@echo "API Health:"
	curl -f http://localhost:8000/health || echo "API not ready"
	@echo "MLflow Health:"
	curl -f http://localhost:5000 || echo "MLflow not ready"
	@echo "Dashboard Health:"
	curl -f http://localhost:8501/_stcore/health || echo "Dashboard not ready"

# Optimized build commands
docker-build-fast:
	docker compose build

docker-build-api:
	docker compose build api

docker-build-dashboard:
	docker compose build dashboard

docker-build-mlflow:
	docker compose build mlflow

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
	@echo "Option A: Local Pipeline (Simple)"
	@echo "  pipeline       - Run simple pipeline (src.run_pipeline.py)"
	@echo "  smoke-test     - Run smoke test"
	@echo "  mlflow-ui      - Start MLflow UI"
	@echo "  promote-staging - Promote model to staging"
	@echo "  rollback-production - Rollback production model"
	@echo ""
	@echo "Option A: Local Pipeline (Enhanced Prefect Orchestration)"
	@echo "  prefect-start  - Start Prefect server"
	@echo "  prefect-setup  - Configure Prefect API URL"
	@echo "  prefect-worker - Start Prefect worker"
	@echo "  prefect-flow   - Run Prefect flow (flows/enhanced_orchestration.py)"
	@echo "  prefect-deploy - Build deployment"
	@echo "  prefect-deploy-apply - Apply deployment"
	@echo "  prefect-deploy-all - Create all scheduled deployments"
	@echo ""
	@echo "Option B: Docker Pipeline (Optimized Reproducible Setup)"
	@echo "  docker-setup   - Complete Docker setup (build + run + pipeline)"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-run     - Start Docker services"
	@echo "  docker-pipeline - Run pipeline in Docker (src.run_pipeline.py)"
	@echo "  docker-smoke-test - Test Docker services"
	@echo "  docker-health-check - Check service health"
	@echo "  docker-logs    - View service logs"
	@echo "  docker-restart - Restart services"
	@echo "  docker-stop    - Stop services"
	@echo "  docker-clean   - Clean up Docker resources"
