#!/bin/sh
# Start script of the ABMEM web container.
# 1. Waits until PostgreSQL accepts connections.
# 2. Creates a secret key on first start and keeps it in a volume.
# 3. Applies the database migrations.
# 4. Loads the starter data when the database is empty.
# 5. Starts the command given by the image (gunicorn by default).
set -eu

: "${DB_HOST:=db}"
: "${DB_PORT:=5432}"
: "${DB_USER:=postgres}"

echo "Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT} ..."
tries=0
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -ge 60 ]; then
        echo "PostgreSQL did not become ready within 120 seconds." >&2
        exit 1
    fi
    sleep 2
done

# Without an explicit DJANGO_SECRET_KEY, Django reads the key from
# DJANGO_SECRET_KEY_FILE (set in the image). Create it on first start.
key_file="${DJANGO_SECRET_KEY_FILE:-/var/lib/abmem/secret_key}"
if [ -z "${DJANGO_SECRET_KEY:-}" ] && [ ! -s "$key_file" ]; then
    python -c "from django.core.management.utils import get_random_secret_key as g; print(g())" > "$key_file"
fi

python manage.py migrate --noinput

if [ "${ABMEM_LOAD_STARTER_DATA:-true}" = "true" ]; then
    python manage.py load_starter_data
fi

echo "ABMEM is starting. Open http://localhost:${ABMEM_PORT:-8000}/ in your browser."
exec "$@"
