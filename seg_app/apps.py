import multiprocessing
import os
from datetime import datetime, timedelta

from django.apps import AppConfig

from seg_app.models import UploadedImage
from seg_app.utils.log import log_info
from seg_app.views import while_process_imgs


class SegAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "seg_app"

    def ready(self):
        # 在 ready() 方法中启动子进程
        print("应用准备就绪")
        process = multiprocessing.Process(target=while_process_imgs)
        # 启动进程
        process.start()
