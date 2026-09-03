FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --use-pep517 -r requirements.txt

COPY . .

RUN python train_and_visualize.py

EXPOSE 5000

CMD ["python", "src/app.py"]
