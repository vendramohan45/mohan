# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /code/

# Create necessary directories and set permissions for Hugging Face (UID 1000)
RUN mkdir -p /code/staticfiles /code/media /code/ml_models
RUN chmod -R 777 /code/

# Run collectstatic and setup
RUN python manage.py collectstatic --no-input
RUN python manage.py migrate
RUN python setup.py

# Expose the port
EXPOSE 7860

# Command to run the application
# Hugging Face Spaces expects the app to run on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "eggdetect.wsgi:application", "--timeout", "1000", "--workers", "1", "--threads", "1"]
