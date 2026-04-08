from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic.base import RedirectView
from users import views as user_views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Direct mappings to avoid include issues on some cloud platforms
    path('login/', user_views.login_view, name='userlogin'),
    path('adminlogin/', user_views.admin_login_view, name='adminlogin'),
    path('signup/', user_views.register_view, name='registration'),
    path('auth/login/', user_views.login_view, name='login'),
    
    # App-level includes as fallback
    path('', include('core.urls')),
    path('users/', include('users.urls')),
    path('detection/', include('detection.urls')),
    
    path('favicon.ico', RedirectView.as_view(url=settings.STATIC_URL + 'favicon.ico')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
