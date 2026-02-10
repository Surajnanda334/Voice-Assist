# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ffmpeg: for audio processing
# build-essential: for compiling python packages
# espeak-ng: for pyttsx3 (if used as fallback)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
