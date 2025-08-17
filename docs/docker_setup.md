# Docker Setup Documentation 

This document provides comprehensive instructions for setting up the Market Master trading prediction system using Docker for full reproducibility.

## 🎯 Overview

- **PostgreSQL Database**: Production-grade database for MLflow backend
- **Complete Service Stack**: API, MLflow, Dashboard, and Prefect orchestration
- **Persistent Storage**: All data and artifacts preserved across restarts
- **Health Monitoring**: Automatic health checks for all services
- **Network Isolation**: Dedicated Docker network for security

## 🚀 Quick Start

### Prerequisites

1. **Docker Desktop** installed and running
2. **Docker Compose** available (usually included with Docker Desktop)
3. **Git** for cloning the repository

### Setup Instructions

```bash
# Build Docker images
make docker-build

# Start all services
make docker-run

# Run the ML pipeline
make docker-pipeline

# Test the setup
make docker-smoke-test
```

## 📊 Services Overview

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| **API** | 8000 | FastAPI model serving | http://localhost:8000/docs |
| **MLflow** | 5000 | Experiment tracking & Model Registry | http://localhost:5000 |
| **Dashboard** | 8501 | Streamlit trading dashboard | http://localhost:8501 |
| **Prefect** | 4200 | Workflow orchestration | http://localhost:4200 |
| **PostgreSQL** | 5432 | Database backend | Internal only |

## 🔧 Service Details

### 1. PostgreSQL Database
- **Image**: `postgres:15-alpine`
- **Database**: `mlflow`
- **User**: `mlflow`
- **Password**: `mlflow_password`
- **Volume**: `postgres_data` (persistent)

### 2. MLflow Tracking Server
- **Backend**: PostgreSQL
- **Artifact Store**: Local filesystem (`./mlruns`)
- **Features**: Experiment tracking, model registry, artifact management

### 3. FastAPI Model Serving
- **Model Loading**: From MLflow Model Registry (`@Production` alias)
- **Fallback**: Local model files in `artifacts/model/`
- **Endpoints**: Health check, predictions, model info

### 4. Streamlit Dashboard
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Features**: Stock prices, predictions, technical analysis
- **Data Source**: API endpoints

### 5. Prefect Orchestration
- **Workflow Management**: Automated pipeline execution
- **Scheduling**: Cron-based deployments
- **Monitoring**: Task-level observability

## 📁 Directory Structure

```
market-master-trading-action-prediction/
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # API service image
├── Dockerfile.dashboard        # Dashboard service image
├── data/                      # Persistent data (mounted)
│   ├── components/            # Stock data CSV files
│   ├── weights/               # Index weights
│   └── monitoring/            # Evidently reports
├── artifacts/                 # Model artifacts (mounted)
│   ├── model/                 # Champion model
│   └── models/                # All candidate models
└── mlruns/                    # MLflow artifacts (mounted)
```

## 🔄 Management Commands

### Service Management
```bash
# Start all services
make docker-run

# Stop all services
make docker-stop

# Restart services
make docker-restart

# View logs
make docker-logs

# Check health
make docker-health-check
```

### Development
```bash
# Build images
make docker-build

# Run pipeline
make docker-pipeline

# Smoke test
make docker-smoke-test

# Clean up
make docker-clean
```

### Database Management
```bash
# Access PostgreSQL
docker-compose exec postgres psql -U mlflow -d mlflow

# Backup database
docker-compose exec postgres pg_dump -U mlflow mlflow > backup.sql

# Restore database
docker-compose exec -T postgres psql -U mlflow -d mlflow < backup.sql
```

## 🧪 Testing the Setup

### API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Component prediction
curl "http://localhost:8000/predict/component?symbol=AAPL"

# Index signal
curl "http://localhost:8000/signal/index?universe=qqq"

# Model info
curl http://localhost:8000/model-info
```

### Dashboard Access
1. Open http://localhost:8501 in your browser
2. Verify real-time data updates
3. Check prediction displays
4. Test interactive features

### MLflow UI
1. Open http://localhost:5000 in your browser
2. Navigate to Experiments tab
3. Check Model Registry
4. Verify artifacts are stored

## 🔍 Troubleshooting

### Common Issues

**1. Services not starting**
```bash
# Check Docker status
docker info

# Check service logs
make docker-logs

# Restart services
make docker-restart
```

**2. Database connection issues**
```bash
# Check PostgreSQL health
docker-compose exec postgres pg_isready -U mlflow

# Check MLflow logs
docker-compose logs mlflow
```

**3. API not responding**
```bash
# Check API health
curl http://localhost:8000/health

# Check API logs
docker-compose logs api
```

**4. Dashboard not loading**
```bash
# Check dashboard logs
docker-compose logs dashboard

# Verify API connectivity
curl http://localhost:8000/health
```

### Performance Optimization

**1. Resource Limits**
```yaml
# Add to docker-compose.yml services
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

**2. Volume Optimization**
```bash
# Use named volumes for better performance
docker volume create market-master-data
```

**3. Network Optimization**
```bash
# Use host networking for local development
docker-compose up --network host
```

## 🔒 Security Considerations

### Production Deployment
1. **Change default passwords** in `.env`
2. **Use secrets management** for sensitive data
3. **Enable SSL/TLS** for external access
4. **Configure firewall rules**
5. **Use private networks**

### Environment Variables
```bash
# Production .env example
POSTGRES_PASSWORD=your_secure_password
MLFLOW_TRACKING_URI=postgresql://user:pass@host:port/db
MODEL_URI=models:/your-model@Production
```

## 📈 Monitoring & Logging

### Health Checks
All services include health checks:
- **API**: HTTP health endpoint
- **PostgreSQL**: Database connectivity
- **MLflow**: Server responsiveness
- **Dashboard**: Streamlit health endpoint

### Logging
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f mlflow
docker-compose logs -f dashboard
```

### Metrics
- **API**: Request/response metrics via FastAPI
- **MLflow**: Experiment tracking metrics
- **PostgreSQL**: Database performance metrics
- **System**: Docker resource usage

## 🚀 Scaling Considerations

### Horizontal Scaling
```yaml
# Scale API service
docker-compose up --scale api=3
```

### Load Balancing
```yaml
# Add nginx service for load balancing
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Data Persistence
```yaml
# Use external volumes
volumes:
  postgres_data:
    external: true
  mlflow_artifacts:
    external: true
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Prefect Documentation](https://docs.prefect.io/)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review service logs using `make docker-logs`
3. Verify Docker and Docker Compose versions
4. Check system resources (CPU, memory, disk space)
5. Ensure internet connectivity for data fetching

### Common Issues

**Data Fetching Issues**: If the pipeline fails to fetch stock data from Yahoo Finance:
- Check your internet connection
- The system will use fallback data for demonstration
- You can manually create sample data files in the `data/components/` directory

**API Startup Issues**: If the API container fails to start:
- The API requires a trained model to be available
- Run the pipeline first: `make docker-pipeline`
- Check MLflow UI to verify model registration

---

**🎉 Your Market Master trading prediction system is now running in a fully reproducible Docker environment!**
