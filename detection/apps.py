from django.apps import AppConfig
import threading
import sys

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        # Pre-loading disabled for Cloud Deployment (Render) to avoid OOM and Timeouts
        pass
