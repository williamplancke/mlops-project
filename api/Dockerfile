FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt /app/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

COPY api /app/api
COPY models /app/models

EXPOSE 8000
CMD ["/opt/venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
