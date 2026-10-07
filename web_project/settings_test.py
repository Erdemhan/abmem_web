"""Settings for the test suite.

Lets pytest run in a fresh checkout or agent worktree without the git-ignored
.env file: DEBUG is enabled so that no secret key is required. Database
credentials still come from the environment or .env when present (local
PostgreSQL with trust authentication needs none). Never use in production.

Each checkout gets its own test database name, derived from its absolute path,
so that parallel test runs in several worktrees do not drop or lock each
other's ``test_abmem`` database.
"""

import hashlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DJANGO_DEBUG", "true")

from .settings import *  # noqa: E402,F401,F403
from .settings import BASE_DIR, DATABASES  # noqa: E402

_checkout_id = hashlib.sha1(str(BASE_DIR).encode("utf-8")).hexdigest()[:8]
DATABASES["default"].setdefault("TEST", {})
DATABASES["default"]["TEST"]["NAME"] = os.environ.get(
    "DJANGO_TEST_DB_NAME", f"test_abmem_{_checkout_id}"
)
# The development database is UTF8, but this server's template databases are
# WIN1252, so a test database cloned from template1 cannot store characters
# such as U+0130. Create it from template0 with UTF8 to match production data.
DATABASES["default"]["TEST"]["CHARSET"] = "UTF8"
DATABASES["default"]["TEST"]["TEMPLATE"] = "template0"

# Checkpoints of runs started by tests never go into the repository. Tests
# override this per test with pytest's tmp_path (tests/conftest.py); this
# default covers code paths that do not see the override.
ABMEM_CHECKPOINT_DIR = Path(tempfile.gettempdir()) / f"abmem_test_checkpoints_{_checkout_id}"
