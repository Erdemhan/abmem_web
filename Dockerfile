# ABMEM web application image (Django + PyTorch CPU).
# Build and start everything with:  docker compose up --build
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DJANGO_SECRET_KEY_FILE=/var/lib/abmem/secret_key \
    ABMEM_SIM_DATA_DIR=/app/sim_data \
    DJANGO_STATIC_ROOT=/app/staticfiles

# build-essential and libpq-dev build psycopg2. postgresql-client provides pg_isready for
# the start script.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of PyTorch first (the default PyPI wheel pulls the CUDA libraries
# and is several GB larger). Its version must match requirements.txt.
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY . .

# Collect the static files at build time. The placeholder key is used only for
# this step, the container gets its own key at start.
RUN DJANGO_SECRET_KEY=build-only DJANGO_DEBUG=false python manage.py collectstatic --noinput \
    && chmod +x docker/entrypoint.sh \
    && useradd --create-home --uid 1000 abmem \
    && mkdir -p /app/checkpoints /app/sim_data /var/lib/abmem \
    && chown -R abmem:abmem /app/checkpoints /app/sim_data /var/lib/abmem

USER abmem

EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
# Simulation and HPO runs are started as separate processes by the web
# process, so a few threaded workers are enough.
CMD ["gunicorn", "web_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "120"]
