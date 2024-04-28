# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: segResponse
Description:
Author: qinlinjian
Datetime: 4/21/2024 5:48 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import json

from django.http import JsonResponse

ERROR = 500
WARNING = 405
SUCCESS = 200

def jsonRes(status,status_code,data={},message=""):
    try:
        data = json.loads(data.to_json())
    except:
        data = json.dumps(data)
    return JsonResponse({'status': status,'data':data, 'message': message}, status=status_code)

if __name__ == '__main__':
    pass
