FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY . .

EXPOSE 8000
