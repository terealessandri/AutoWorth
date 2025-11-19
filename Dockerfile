# -------- Base image --------
FROM python:3.12-slim

# Logs amigables y sin .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# -------- Install deps --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy code --------
# Copiamos solo lo necesario (ajusta si tu código está en otras rutas)
COPY app.py ./app.py
COPY 02-src ./02-src

# -------- Copy trained model (artifact) --------
# El job "Build, Test, and Deploy" descarga esto ANTES de la build
COPY 05-artifacts/models ./05-artifacts/models

# -------- Expose & run --------
# Usa 8080 para calzar con el curl del workflow
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
