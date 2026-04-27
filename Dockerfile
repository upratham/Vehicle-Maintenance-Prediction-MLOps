# ─── Stage 1: build the React/Vite SPA ────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ─── Stage 2: Python backend serving the SPA + API ────────────────────────
FROM python:3.12-slim-bookworm
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
COPY preprocessor_obj /app/preprocessor_obj

# Drop the built SPA where FastAPI's StaticFiles can pick it up
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

EXPOSE 8000
CMD ["python3", "app.py"]
