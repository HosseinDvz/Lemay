FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required OS dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy code and model
COPY . /app
COPY ./model /app/model

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Set HF cache to avoid external downloads
ENV TRANSFORMERS_CACHE=/app/model


###### Entry point

# For AWS image
# CMD ["python", "worker.py"]

# For local image with the possiblily changing the number of workers at the run time
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker main:app --workers ${WORKERS:-4} --bind 0.0.0.0:8000 --preload"]

