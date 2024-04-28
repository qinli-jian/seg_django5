# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: log
Description:
Author: qinlinjian
Datetime: 4/21/2024 12:23 PM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import logging
import os

from seg_django_5 import settings
from datetime import datetime
# 配置日志记录器
logging.basicConfig(filename=os.path.join(settings.LOG_PATH,str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+'.log'), level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(info):
    logging.info(info)

def log_error(error):
    logging.error(error)

def some_function():
    try:
        # 模拟一个操作
        result = 10 / 0
    except Exception as e:
        # 记录错误日志
        logging.error("An error occurred: %s", str(e), exc_info=True)

def main():
    # 记录信息日志
    logging.info("Starting the program...")
    some_function()
    # 记录信息日志
    logging.info("Program finished successfully.")

if __name__ == '__main__':
    pass
