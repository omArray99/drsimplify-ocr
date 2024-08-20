from django.urls import path
from . import views


from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # path('', views.upload_image, name='upload_image'),
    path('', views.main, name="home_page"),
    path('upload_image/', views.upload_image, name='upload_image'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
