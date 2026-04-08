from django.apps import AppConfig

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        # Startup loading disabled to prevent 502/OOM errors on 512MB RAM servers.
        # AI Engine will now initialize on the first request.
        pass
