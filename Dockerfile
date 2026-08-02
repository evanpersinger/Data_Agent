# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv (Python package manager)
RUN pip install --no-cache-dir uv

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Copy application code
COPY . .

# Install dependencies using uv
# This installs the project and all its dependencies from pyproject.toml
RUN uv pip install --system -e .

# Create necessary directories
RUN mkdir -p raw_data clean_data kaggle_data plots

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the data agent CLI. Swap for the HTTP server with:
#   CMD ["uvicorn", "--app-dir", "backend", "server:app", "--host", "0.0.0.0", "--port", "8017"]
CMD ["python", "backend/data_agent.py"]
