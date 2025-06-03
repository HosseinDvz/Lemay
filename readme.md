# 🧠 Website Content Classifier API

This repository provides a production-ready FastAPI application for zero-shot classification of website content using Hugging Face's `facebook/bart-large-mnli` model. It supports both local and Dockerized deployment, with optimized CPU/GPU inference and parallel request handling.

---

## 🚀 Features

- 🔎 **Zero-Shot Classification** via Hugging Face Transformers  
- 🧪 **FastAPI REST API** endpoint at `/classify`  
- ⚡ **Parallel inference** support using `httpx` and `asyncio`  
- 🐳 **Docker-ready** with configurable Gunicorn workers  
- 📁 Supports **pre-downloaded model caching** for faster container startup  

---

## 🧩 Endpoint

### `POST /classify`

#### Request Body
```json
{
  "website": "example.com",
  "content": "Latest updates on AI and deep learning trends..."
}
```

#### Response
```json
{
  "label": "technology"
}
```

---

## 🛠️ Installation

### Local (Python 3.10+)

```bash
pip install -r requirements.txt

# Change the number of workers based on your hardware
gunicorn -k uvicorn.workers.UvicornWorker main:app --workers 10 --bind 0.0.0.0:8000 --preload
```

### Docker
```bash
docker build -t website-classifier .
docker run -p 8000:8000 website-classifier
```

### Optional: Custom Worker Count
```bash
docker run -e WORKERS=6 -p 8000:8000 website-classifier
```

---

## ⚙️ Pre-Downloaded Model

To avoid downloading the model during runtime:

It's highly recommended to cache the model locally in advance, especially when using parallel inference. Parallel execution can trigger multiple simultaneous model load requests to the Hugging Face Hub, which may cause you to exceed rate limits and result in failed or incomplete responses.

You can manually download and cache the model as follows:

Cache the model manually:
```python
from transformers import AutoModel, AutoTokenizer

AutoModel.from_pretrained("facebook/bart-large-mnli", cache_dir="./model")
AutoTokenizer.from_pretrained("facebook/bart-large-mnli", cache_dir="./model")
```

## 🧪 Testing the Application

After running the app, you may use the provided Jupyter notebook to test the classification API.

A folder containing example CSV files (`scrapingResults/`) is included in the repository. Each file contains test data with website names and corresponding content.

The notebook demonstrates:
- How to send classification requests to the API
- How to perform parallel inference using `asyncio` and `httpx`
- Timing comparisons between sequential and parallel inference

To launch the notebook:
```bash
jupyter notebook
```

---

## 📋 Notes
- The `worker.py` script is intended for use when deploying the image on AWS using ECS Fargate for inference.  
- This setup has been tested and is confirmed to be working as expected in that environment.

- For maximum performance, it's recommended to run the app **outside Docker**
- The model can be easily swapped for another zero-shot model or custom classifier.

---

## 🧼 License

MIT License. See [LICENSE](./LICENSE) file for details.
