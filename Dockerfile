FROM python:3.12-slim AS hello
WORKDIR /app
COPY hello_world.py .
CMD ["python", "hello_world.py"]

FROM hello AS lab
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY scripts scripts
COPY web web
COPY docs docs
COPY models/registry.json models/registry.json
COPY models/licenses models/licenses
RUN python scripts/fetch_models.py
CMD ["python", "app.py", "--host", "0.0.0.0"]
