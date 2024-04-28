# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: form
Description:
Author: qinlinjian
Datetime: 4/10/2024 2:28 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"
from django import forms
from .models import UploadedImage
#
# class ImageUploadForm(forms.ModelForm):
#     # 内部类Meta：
#     # 在表单类中，我们通常会定义一个内部类Meta，用于指定一些元数据，
#     # 如使用哪个模型来创建表单，以及应该包含哪些字段。
#     # model = UploadedImage：指定了表单应该基于的模型，即UploadedImage模型。
#     # fields = ['image', 'image_name']：指定了应该在表单中包含哪些字段，这里包括
#     # image和image_name字段。
#     class Meta:
#         model = UploadedImage
#         fields = ['image', 'image_name']




if __name__ == '__main__':
    pass
