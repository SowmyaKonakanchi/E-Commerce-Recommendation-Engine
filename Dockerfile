FROM python:3.11-slim

WORKDIR /app

# Install build tools required by scikit-surprise
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train models at build time
RUN python train_and_visualize.py

EXPOSE 5000

CMD ["python", "src/app.py"]
