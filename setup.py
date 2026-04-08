#!/usr/bin/env python
"""
EggDetect AI - Django Setup Script
This script initializes the database and creates the default admin user.
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eggdetect.settings')
django.setup()

from django.contrib.auth.hashers import make_password
from users.models import User

print("=" * 60)
print("  EggDetect AI - Database Setup")
print("=" * 60)
print()

# Create default admin user
if not User.objects.filter(username='admin').exists():
    User.objects.create(
        username='admin',
        email='admin@eggdetect.com',
        password=make_password('admin123'),
        role='admin',
        is_superuser=True,
        is_staff=True
    )
    print("✓ Default admin user created")
    print("  Username: admin")
    print("  Password: admin123")
else:
    print("✓ Admin user already exists")

print()
print("=" * 60)
print("  Setup complete! Run: python manage.py runserver")
print("=" * 60)
