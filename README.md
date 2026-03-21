# 🚗 Vehicle Maintenance Prediction
### An End-to-End Machine Learning Pipeline with MLOps

---
**Live Demo (Render):** https://vehicle-maintenance-prediction-mlops.onrender.com
## 📌 Project Overview

**Vehicle Maintenance Prediction** is a production-grade machine learning application that predicts vehicle maintenance needs based on historical data. The project demonstrates a complete MLOps lifecycle — from raw data ingestion to a deployed, continuously integrated web application — highlighting real-world engineering practices used in industry.

> 🎯 **Goal:** Predict whether a vehicle requires maintenance, enabling proactive servicing and reducing breakdown risks.

---

## 🏗️ System Architecture

```
Raw Data (MongoDB Atlas)
        │
        ▼
┌─────────────────┐
│  Data Ingestion  │ ◄── MongoDB Atlas Connection
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Data Validation      │ ◄── Schema YAML Config
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────┐
│  Data Transformation      │ ◄── Feature Engineering
└────────┬─────────────────┘
         │
         ▼
┌───────────────────┐
│   Model Trainer    │ ◄── Sklearn / Custom Estimator
└────────┬──────────┘
         │
         ▼
┌──────────────────────┐
│  Model Evaluation     │ ◄── Threshold: 0.02 drift check
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│   Model Pusher        │ ◄── AWS S3 Model Registry
└────────┬─────────────┘
         │
         ▼
┌────────────────────────────┐
│  Flask Web Application      │ ◄── Prediction + Training Routes
└────────┬───────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Docker + EC2 + GitHub Actions    │ ◄── CI/CD Pipeline
└──────────────────────────────────┘
```

---

## 🛠️ Technologies & Services Used

### 🐍 Languages & Frameworks
| Technology | Purpose |
|---|---|
| Python 3.10 | Core programming language |
| Flask | Web application framework |
| Scikit-learn | Machine learning model training & evaluation |
| Pandas / NumPy | Data manipulation and analysis |

### ☁️ Cloud & Infrastructure
| Service | Purpose |
|---|---|
| **AWS EC2** | Cloud server hosting (Ubuntu 24.04, T2 Medium) |
| **AWS S3** | Model registry & artifact storage |
| **AWS ECR** | Docker image repository |
| **AWS IAM** | Secure access management with scoped policies |
| **MongoDB Atlas** | Cloud NoSQL database for raw data storage |

### ⚙️ MLOps & DevOps
| Tool | Purpose |
|---|---|
| **Docker** | Containerization for consistent deployments |
| **GitHub Actions** | CI/CD pipeline automation |
| **Self-hosted Runner** | EC2-based GitHub Actions runner |
| **Conda** | Environment & dependency management |

### 📦 Project Structure & Tooling
| Tool | Purpose |
|---|---|
| `setup.py` + `pyproject.toml` | Local package installation |
| `requirements.txt` | Dependency management |
| Custom Logger | Centralized logging across all components |
| Custom Exception Handler | Structured error handling |

---

## 🔄 ML Pipeline Components

### 1️⃣ Data Ingestion
- Connects to **MongoDB Atlas** using a secure connection string via environment variables
- Fetches raw data in key-value format and transforms it into a structured DataFrame
- Outputs ingestion artifacts for downstream components

### 2️⃣ Data Validation
- Validates incoming data against a **schema YAML config** (column names, data types, value ranges)
- Detects data drift and missing features before they silently corrupt the model

### 3️⃣ Data Transformation
- Applies feature engineering pipelines defined during EDA
- Uses a custom `estimator.py` in the entity layer for reusable preprocessing logic

### 4️⃣ Model Trainer
- Trains and evaluates ML models with cross-validation
- Saves the best model as a serialized artifact

### 5️⃣ Model Evaluation
- Compares the newly trained model against the production model stored in **AWS S3**
- Promotes the new model only if performance improves by more than the threshold score (`0.02`)

### 6️⃣ Model Pusher
- Pushes the accepted model to the **AWS S3 model registry** (`my-model-mlopsproj` bucket)
- Maintains a versioned model history for rollback capability

---

## 🚀 CI/CD Pipeline

```
Developer pushes code to GitHub
            │
            ▼
  GitHub Actions workflow triggered (aws.yaml)
            │
            ▼
  Build Docker image → Push to AWS ECR
            │
            ▼
  Self-hosted EC2 runner pulls latest image
            │
            ▼
  Container deployed → App live on EC2:5080
```

The pipeline requires the following **GitHub Secrets** to be configured:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS programmatic access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_DEFAULT_REGION` | Target AWS region (`us-east-1`) |
| `ECR_REPO` | ECR repository URI |

---

## 🗂️ Project Structure

```
vehicle-maintenance-prediction/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/
│   │   ├── mongo_db_connections.py
│   │   └── aws_connection.py
│   ├── entity/
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   ├── data_access/
│   ├── aws_storage/
│   ├── pipeline/
│   │   └── training_pipeline.py
│   ├── utils/
│   │   └── main_utils.py
│   └── constants/
│       └── __init__.py
│
├── config/
│   └── schema.yaml
├── notebook/
│   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   └── mongoDB_demo.ipynb
├── static/
├── templates/
├── app.py
├── demo.py
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Dockerfile
├── .dockerignore
└── .github/
    └── workflows/
        └── aws.yaml
```

---

## ⚙️ Local Setup & Installation

### Prerequisites
- Conda installed
- MongoDB Atlas account with a cluster
- AWS account with S3, EC2, and ECR access

### 1. Clone the Repository
```bash
git clone https://github.com/upratham/Vehicle-Maintenance-Prediction-MLOps.git
cd Vehicle-Maintenance-Prediction-MLOps
```

### 2. Create & Activate Conda Environment
```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip list  # Verify local packages are installed
```

### 4. Set Environment Variables

**Bash (Mac/Linux):**
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

**PowerShell (Windows):**
```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster.mongodb.net/"
$env:AWS_ACCESS_KEY_ID = "your_access_key"
$env:AWS_SECRET_ACCESS_KEY = "your_secret_key"
```

### 5. Run the Application
```bash
python app.py
```

Visit `http://localhost:5080` in your browser.

---

## 🌐 Application Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page with prediction form |
| `/predict` | POST | Returns maintenance prediction |
| `/training` | GET | Triggers the full ML training pipeline |

---

## 📊 Key Skills Demonstrated

| Skill Area | What I Built |
|---|---|
| **Machine Learning** | End-to-end pipeline with validation, transformation, training, and evaluation |
| **MLOps** | Automated model promotion with drift threshold checks and S3 model registry |
| **Cloud Engineering** | Multi-service AWS setup (IAM, S3, EC2, ECR) with least-privilege access |
| **Database Engineering** | NoSQL data management with MongoDB Atlas, cloud data push/pull |
| **DevOps / CI-CD** | Fully automated GitHub Actions pipeline with Docker and self-hosted EC2 runner |
| **Software Engineering** | Modular codebase with custom logging, exception handling, config entities, and artifacts |
| **Data Science** | EDA and feature engineering documented in Jupyter notebooks |

---

## 👥 Authors

| Name | Email |
|---|---|
| **Prathamesh Uravane** | upratham2002@gmail.com |
| **Sankeerth B** | sankeerth.b@example.com |
| **Claude B** | claude.b@example.com |

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

⭐ *If you found this project helpful or interesting, please consider giving it a star!*
