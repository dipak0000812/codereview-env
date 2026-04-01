FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify data folder exists at build time
RUN python -c "from pathlib import Path; assert Path('data/task1').exists(), 'data/task1 missing'; assert Path('data/task2').exists(), 'data/task2 missing'; assert Path('data/task3').exists(), 'data/task3 missing'"

# Environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run with uvicorn (single worker for HF Spaces compatibility)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
