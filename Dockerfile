# -------- Base image --------
FROM python:3.12-slim

# Optional: nicer logs & no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# -------- Install deps --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy code + model artifacts --------
# (CI workflow downloads 05-artifacts/models/ before build; locally you already have them)
COPY . .

# -------- Expose & run --------
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
