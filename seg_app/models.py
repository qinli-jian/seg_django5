import datetime

from django.db.models.signals import pre_save
from django.dispatch import receiver
from mongoengine import Document, StringField, DateTimeField, DictField, ListField,IntField
from pytz import timezone as pytz_timezone
from django.utils import timezone
# 获取北京时区
beijing_tz = pytz_timezone('Asia/Shanghai')

class UploadedImage(Document):
    image_name = StringField(max_length=100)  # 使用 StringField 字段保存图片名称
    depthimg_name = StringField(max_length=100,default="") # 可以缺省 new
    uploaded_at = DateTimeField(default=timezone.now().astimezone(beijing_tz))  # 使用 DateTimeField 字段保存上传时间
    processed_image_name = StringField(max_length=100)
    palette_dict = DictField()
    segments = ListField(DictField())
    state = IntField()

    def clean(self):
        # 将palette_dict中的非字符串键转换为字符串类型
        self.palette_dict = {str(k): v for k, v in self.palette_dict.items()}

    # 定义信号接收器
# @receiver(pre_save, sender=UploadedImage)
# def update_uploaded_at(sender, instance, **kwargs):
#     instance.uploaded_at = datetime.datetime.now()
