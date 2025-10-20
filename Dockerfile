FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git jq python3-dev gcc libffi-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install torch CPU wheels first (if using transformer text gen)
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.3.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# create output folder for generated compose files
RUN mkdir -p /app/out

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
