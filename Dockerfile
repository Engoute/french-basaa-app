# Use an official NVIDIA base image compatible with PyTorch 2.2+
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up a non-interactive environment
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, git (for pip git installs), and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    unzip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Link python3.10 to the 'python' command
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set the working directory
WORKDIR /app

# Copy and install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY ./app /app/app

# Expose the port the app will run on
EXPOSE 8000

# The command to run the application when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
