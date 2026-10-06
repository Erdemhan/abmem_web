"""Settings for the test suite.

Lets pytest run in a fresh checkout or agent worktree without the git-ignored
.env file: DEBUG is enabled so that no secret key is required. Database
credentials still come from the environment or .env when present (local
PostgreSQL with trust authentication needs none). Never use in production.
"""

import os

os.environ.setdefault("DJANGO_DEBUG", "true")

from .settings import *  # noqa: E402,F401,F403
