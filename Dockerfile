# ------------------------------------------------------
# RocketPy Development Container
# ------------------------------------------------------

FROM python:3.13-slim

LABEL maintainer="Tanaya Bhoyar"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system packages required by RocketPy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install RocketPy dependencies
RUN pip install -e ".[tests]"

# Default command
CMD ["python"]