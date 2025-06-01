FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy app code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run using Gunicorn + Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--workers", "4", "--bind", "0.0.0.0:8000"]
