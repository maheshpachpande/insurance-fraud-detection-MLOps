
FROM python:3.12-slim AS build

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc/pyo files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy only the source code (excluding ignored dirs via .dockerignore)
COPY . /app/


# Expose API port
EXPOSE 8080

# Start 
CMD ["python", "app.py"]
