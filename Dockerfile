FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install as a package to create the 'server' entry point
RUN pip install -e .

ENV PORT=7860
EXPOSE 7860

CMD ["server"]
