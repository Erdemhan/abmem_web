from django.apps import AppConfig


class AbmemConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "abmem"

    def ready(self) -> None:
        """Connect the signal that deletes checkpoint files with their rows."""
        from abmem.services.simulation import checkpoints  # noqa: F401
