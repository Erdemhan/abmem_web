"""ASGI entry point for the web_project Django site.

Exposes the standard Django ASGI handler. The project serves HTTP only, and
the UI obtains simulation progress through HTTP polling.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")

application = get_asgi_application()
