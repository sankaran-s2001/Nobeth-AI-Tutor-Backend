import os
from celery import Celery

# Set default Django settings module for 'celery' program
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tutor_backend.settings')

app = Celery('tutor_backend')

# Load task configurations from settings.py using 'CELERY_' namespace
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover task files (tasks.py) in all installed Django apps
app.autodiscover_tasks()
