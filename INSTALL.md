# Installing ABMEM

There are two ways to install ABMEM.

| | Docker (recommended) | Development install |
|---|---|---|
| For | Using the simulator | Changing the code, running the tests |
| You install | Docker Desktop and Git | Python 3.11, PostgreSQL 16 and Git |
| Database | Created automatically in a container | You create it |
| Time | About 10 minutes on the first start | About 20 minutes |

Both give the same web interface at the end.

---

## A. Docker (recommended)

### A1. Install Docker Desktop

**Windows 10/11**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/ and run the installer.
2. Keep "Use WSL 2 instead of Hyper-V" selected.
3. Restart the computer when the installer asks for it.
4. Start Docker Desktop and wait until it shows "Engine running".

If Docker Desktop reports that WSL is missing or outdated, open a terminal (PowerShell) and run `wsl --update`, then start Docker Desktop again. If it reports that virtualization is disabled, enable "Intel VT-x" or "AMD-V/SVM" in the BIOS/UEFI settings of the computer.

**macOS**: download Docker Desktop for your chip (Apple silicon or Intel), drag it to Applications, start it.

**Linux**: install Docker Engine and the Docker Compose plugin from the package instructions on https://docs.docker.com/engine/install/.

Check in a terminal:

```
docker --version
docker compose version
```

Disk space: the image needs about 2.5 GB and the database grows with the number of runs. Docker Desktop stores its data on the system drive by default. If that drive has less than 10 GB free, move it first in Docker Desktop under Settings > Resources > Advanced > Disk image location.

### A2. Install Git and get the project

Install Git from https://git-scm.com/downloads (Windows: keep the default options). Then, in a terminal, go to the folder where the project should live and run:

```
git clone https://github.com/Erdemhan/abmem_web.git
cd abmem_web
```

Without Git, download the ZIP from the "Code" button on https://github.com/Erdemhan/abmem_web and unpack it. Updating is then a new download instead of `git pull`.

### A3. Start

In the project folder (the one that contains `docker-compose.yml`):

```
docker compose up --build -d
```

The first start downloads Python, PyTorch and PostgreSQL and builds the image. This takes a few minutes. When the command returns, the application needs a few more seconds to create the database. Open:

**http://localhost:8000/**

On the first start the database is created, all migrations are applied and the starter data is loaded: 8 resources and 3 scenario templates (Mixed, Only Fossil, Only Renewable) with their library agents.

### A4. Check that it works

1. Open "Templates" in the left menu. The three scenarios are listed.
2. Run "Only Renewable". After about 10 to 20 seconds the run is FINISHED and its analysis page opens.

If the page does not open, look at the log:

```
docker compose logs web
```

The last lines should be "Starter data loaded ..." (first start only) and "Listening at: http://0.0.0.0:8000".

### A5. Everyday use

| Task | Command (in the project folder) |
|---|---|
| Start | `docker compose up -d` |
| Stop (all data is kept) | `docker compose down` |
| Update to the latest version | `git pull` then `docker compose up --build -d` |
| Show the log | `docker compose logs -f web` |
| Back up the database | `docker compose exec db pg_dump -U postgres -Fc abmem > abmem.dump` |
| Delete everything, including all runs | `docker compose down -v` |

Docker Desktop must be running for these commands. Migrations are applied automatically on every start, so `git pull` plus `docker compose up --build -d` is the whole update.

### A6. Optional settings

Without any configuration the defaults are used. To change them, copy `.env.docker.example` to `.env` in the project folder and edit it:

- `ABMEM_PORT`: port of the web page (default 8000). Change it if another program uses 8000.
- `DB_PASSWORD`: password of the container database.
- `ABMEM_LOAD_STARTER_DATA=false`: start with an empty database.

Then run `docker compose up -d` again. The application is reachable only from this computer. `docs/docker.md` explains how to open it to the network and where Docker keeps the data.

---

## B. Development install

Use this to change the code or run the test suite.

### B1. Requirements

- Python 3.11 (python.org installer, or a conda environment: `conda create -n abmem python=3.11`)
- PostgreSQL 16 (https://www.postgresql.org/download/). Remember the password of the `postgres` user set during installation.
- Git

### B2. Get the project and install the packages

```
git clone https://github.com/Erdemhan/abmem_web.git
cd abmem_web
python -m venv .venv
```

Activate the environment:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (Git Bash): `source .venv/Scripts/activate`
- macOS/Linux: `source .venv/bin/activate`

With conda, use `conda activate abmem` instead of the venv.

Install PyTorch first, then the rest:

```
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

For an NVIDIA GPU, install the CUDA build of torch 2.4.0 that matches the driver instead (see https://pytorch.org/get-started/previous-versions/). Runs do not need a GPU.

### B3. Configure

Copy `.env.example` to `.env` in the project folder and set at least:

```
DJANGO_DEBUG=true
DB_PASSWORD=<password of the postgres user>
```

`DJANGO_SECRET_KEY` may stay empty while `DJANGO_DEBUG=true`. With `DJANGO_DEBUG=false` a key is required. Generate one with:

```
python -c "from django.core.management.utils import get_random_secret_key as g; print(g())"
```

### B4. Create the database

Create an empty UTF-8 database named `abmem`, with pgAdmin or in a terminal:

```
createdb -U postgres -E UTF8 -T template0 abmem
```

Then create the tables and load the starter data:

```
python manage.py migrate
python manage.py load_starter_data
```

### B5. Run

```
python manage.py runserver
```

Open http://127.0.0.1:8000/ in a browser. The development server reloads itself when a file changes. After pulling a new version, run `python manage.py migrate` before starting it, because new versions can add database migrations.

### B6. Tests

```
python -m pytest
```

The tests create their own temporary database, so the PostgreSQL user needs the CREATEDB right. It is missing only for users you created yourself. Grant it once with `ALTER ROLE <user> CREATEDB;` in pgAdmin or psql. The Docker smoke test (`docker/smoke_test.sh`) runs on GitHub Actions after every push.

---

## Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `failed to connect to the docker API` | Docker Desktop is not running. Start it and wait for "Engine running". |
| `port is already allocated` / page of another program | Port 8000 is in use. Set `ABMEM_PORT=8001` in `.env` (A6) and open http://localhost:8001/. |
| Docker reports `read-only file system` or `input/output error` | The drive with Docker's disk image is full. Free space or move the disk image (A1). |
| `exec docker/entrypoint.sh: no such file or directory` | The files were checked out with Windows line endings by an old Git setup. Run `git config core.autocrlf input`, delete the folder, clone again. |
| `DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is false` (development install) | Set `DJANGO_DEBUG=true` in `.env`, or set a secret key (B3). |
| `password authentication failed for user "postgres"` | `DB_PASSWORD` in `.env` does not match the PostgreSQL password. |
| `database "abmem" does not exist` | Create it (B4). |
| `permission denied to create database` when running tests | Grant CREATEDB (B6). |
| A run waits after the first period | The scenario is in "Period by period" mode. Use the Continue button on the watch page, or set "Only result" in the scenario. |
