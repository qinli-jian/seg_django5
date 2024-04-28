# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: urls
Description:
Author: qinlinjian
Datetime: 4/10/2024 2:07 PM
Product: PyCharm
-----------------------------------------------------------
"""

from django.urls import path

from . import views

urlpatterns = [
    path("uploadimg/", views.upload_image, name="index"),
    path("findimgbyname/", views.find_img_byname, name="findimgbyname"),
    path("findimgbyperiod/", views.find_img_byperiod, name="findimgbyperiod"),
]