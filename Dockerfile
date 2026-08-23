FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by Python packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, data, and persisted indexes.
COPY . .

EXPOSE 8000 8501

# The actual command is provided by docker-compose.yml.
CMD ["python", "-c", "print('Use docker compose to start the API or frontend service.')"]