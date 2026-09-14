FROM python:3.11-slim

WORKDIR /app

# Install dependencies separately for layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || pip install --no-cache-dir -e .

# Copy source
COPY agentbench/ agentbench/
COPY evals/ evals/

EXPOSE 8080

CMD ["agentbench", "serve", "--host", "0.0.0.0", "--port", "8080"]
