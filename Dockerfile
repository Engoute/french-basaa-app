# Use the official NVIDIA development image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip libsndfile1 git && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install gdown separately to ensure it's available
RUN pip install --no-cache-dir gdown

# Install all other packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app .

# The command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
