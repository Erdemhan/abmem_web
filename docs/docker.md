# Running ABMEM with Docker

This is the easiest way to run ABMEM on a computer that has no Python or
PostgreSQL. Docker runs the web application and its database in two
containers. The first start installs everything, creates the database, applies
the migrations and loads the starter data.

## 1. Install Docker

- Windows and macOS: install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  and start it once. On Windows, accept the WSL 2 setup when Docker Desktop
  asks for it.
- Linux: install Docker Engine and the Docker Compose plugin.

Disk space: the image needs about 2 GB and the database grows with the
number of runs. Docker Desktop keeps its data on the system drive by default.
If that drive is nearly full, Docker stops with "read-only file system" or
"input/output error" messages. Free some space or move the disk image in
Docker Desktop under Settings > Resources > Advanced > Disk image location.

## 2. Get the project

Clone the repository or unpack a copy, then open a terminal in its folder
(the folder that contains `docker-compose.yml`).

## 3. Start

```bash
docker compose up --build -d
```

The first start downloads and builds the image and takes a few minutes. Later
starts take a few seconds. Then open http://localhost:8000/ in a browser.

The starter data is loaded only into an empty database. It contains the
resource list and three scenario templates (Mixed, Only Fossil, Only
Renewable) with their library agents.

## Everyday commands

| Task | Command |
|---|---|
| Start | `docker compose up -d` |
| Stop (data is kept) | `docker compose down` |
| Show the log | `docker compose logs -f web` |
| Update after a code change | `docker compose up --build -d` |
| Run a management command | `docker compose exec web python manage.py <command>` |
| Back up the database | `docker compose exec db pg_dump -U postgres -Fc abmem > abmem.dump` |
| Delete everything, including all data | `docker compose down -v` |

Migrations are applied automatically at every start, so pulling a new version
and running `docker compose up --build -d` is enough to update.

## Settings

Defaults work without any configuration. To change them, copy
`.env.docker.example` to `.env` next to `docker-compose.yml` and edit the
values:

- `ABMEM_PORT`: port on this computer (default 8000).
- `DB_PASSWORD`: password of the container database.
- `ABMEM_LOAD_STARTER_DATA`: `false` starts with an empty database.
- `DJANGO_SECRET_KEY`: leave empty and the container creates and keeps its
  own key.

The application is reachable only from this computer. To open it to the
network, change the port mapping in `docker-compose.yml` to `"8000:8000"`
and add the computer's host name or address to `DJANGO_ALLOWED_HOSTS`.

## Where the data is

Docker volumes keep the data between restarts:

- `db-data`: the PostgreSQL database (scenarios, runs, results).
- `checkpoints`: saved policies of finished runs.
- `sim-data`: the JSON offer export of every run.
- `app-state`: the generated secret key.

`docker compose down` keeps the volumes. `docker compose down -v` deletes
them.

## Notes

- The image uses the CPU build of PyTorch. Runs do not need a GPU.
- A run in "Period by period" mode waits on its watch page for the Continue
  button after every period. Use "Only result" to run to the end without
  stopping.
