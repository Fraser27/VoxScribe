FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    ffmpeg \
    git \
    curl
# Upgrade pip and install build tools
RUN pip3 install --break-system-packages --upgrade pip setuptools wheel
# Install numpy first (required by sox and other packages)
RUN pip3 install --break-system-packages --no-cache-dir numpy>=1.20.0

RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models logs transcriptions

EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
