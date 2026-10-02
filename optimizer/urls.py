from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("black-litterman/", views.black_litterman, name="black_litterman"),
    path("export/<str:run>/<str:kind>.csv", views.export_csv, name="export_csv"),
    path("plotly.js", views.plotly_js, name="plotly_js"),
]
