FROM python:3.11-slim

WORKDIR /app

# Install git (required for pip install from git+https URLs)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify data folder exists
RUN python -c "from pathlib import Path; assert Path('data/task1').exists(), 'data/task1 missing'"

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
