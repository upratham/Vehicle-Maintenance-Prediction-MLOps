# Multi-Model Training Pipeline Refactor Report

## Goal Completed
Enabled the training system to support 3 model profiles with separate:
- dataset/collection routing
- schema + target handling
- preprocessing object persistence
- artifacts and model metadata

## New Files Added
- config/pipeline_profiles.yaml
- config/schema_cars_hyundai.yaml
- config/schema_engine_data.yaml
- config/model_cars_hyundai.yaml
- config/model_engine_data.yaml
- MULTI_MODEL_PIPELINE_REPORT.md

## Core Refactor Summary

### 1) Profile-driven orchestration
Updated src/pipline/training_pipeline.py to:
- load profiles from config/pipeline_profiles.yaml
- switch config context per profile using _set_profile_context(...)
- run pipeline for selected profiles in one call
- support optional refresh_collections per selected profile collections
- return per-profile run summary including trained model path, preprocessor path, metrics, and S3 model path

### 2) Separate artifacts per model/profile
Refactored src/entity/config_entity.py:
- artifact root is now artifact/<profile_name>/<timestamp>
- model registry path is profile-specific
- S3 key path is profile-specific: model-registry/<profile_name>/model.pkl
- baselines.json is profile-specific

### 3) Separate preprocessing object per model/profile
Refactored src/entity/config_entity.py:
- preprocessing object is now saved at preprocessor_obj/<profile_name>/preprocessing.pkl

### 4) Per-profile schema + target support
Updated:
- src/components/data_validation.py to read schema path from config
- src/components/data_transformation.py to read schema path from config and resolve target dynamically

Target resolution in data transformation now supports:
- exact match
- normalized match (case/space/punctuation insensitive)
- fallback to schema target or last dataset column with warning logs

### 5) Multi-dataset ingestion support
Updated:
- src/data_access/proj1_data.py to accept explicit database_name
- src/components/data_ingestion.py to pass database_name and collection_name from profile config

### 6) Model metadata separation
Updated:
- src/components/model_trainer.py to write baselines per profile artifact path
- src/components/model_evaluation.py to include profile + metrics in artifact
- src/components/model_pusher.py to write model_registry.json per profile artifact path

### 7) Artifact dataclasses enhanced
Updated src/entity/artifact_entity.py with optional profile_name and metrics fields where needed for traceability.

## Validation Performed
- Static error check passed for all changed files (except pre-existing TensorFlow import resolution warning in model_trainer).
- Runtime smoke-check verified:
  - profiles loaded correctly
  - artifact paths separated per profile
  - preprocessing paths separated per profile
  - S3 key paths separated per profile
  - profile-specific schema and target values wired into config objects

## Notes
- In pipeline_profiles.yaml you requested:
  - cars_hyundai target_column: "hyundai cars"
  - engine_data target_column: "engine condition"
- Dataset columns may differ in exact casing/wording; dynamic target resolution now handles this robustly.

## How To Run
- single profile run (example):
  - TrainPipeline().run_pipeline(profile_names=["engine_data"], refresh_collections=False)
- all profiles:
  - TrainPipeline().run_pipeline(refresh_collections=False)
- refresh collections before training selected profiles:
  - TrainPipeline().run_pipeline(refresh_collections=True, profile_names=["vehicle_maintenance", "cars_hyundai", "engine_data"])
