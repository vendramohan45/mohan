from django.apps import AppConfig
import threading
import gc

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        """Pre-load Xception model during startup to prevent analysis timeouts"""
        import os
        # Only load if we are the main process (not reloader)
        if os.environ.get('RUN_MAIN') != 'true':
            def startup_warmup():
                try:
                    from .views import get_model
                    print(">>>> [SYSTEM] AI Engine Init Started...")
                    get_model('xception')
                    print(">>>> [SYSTEM] AI Engine Ready for Production.")
                except Exception as e:
                    print(f">>>> [SYSTEM] AI Warmup skipped: {e}")
                finally:
                    gc.collect()

            # Run in a separate thread so Django boots fast
            threading.Thread(target=startup_warmup, daemon=True).start()
