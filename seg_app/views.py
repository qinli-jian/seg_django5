import json
import os
from datetime import timedelta

from django.shortcuts import render

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from seg_app.models import UploadedImage
from seg_app.utils.snowflake_idgen import IdWorker
from seg_django_5 import settings
from seg_app.seg.predict_img import SegImg
from seg_app.utils.log import *

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

idgen = IdWorker(1, 1, 0)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_name = request.FILES.get('image').name  # 获取上传图片的名称
        # name = request.POST.get("image_name")
        _ = save_image(request.FILES.get('image'), image_name)
        if _ == 0:
            return JsonResponse({'status': 'error', 'message': 'save error'}, status=500)

        try:
            # 处理图片，还要把处理之后的所有数据都输出，label、score、bbox:[]等
            # processed_image_name, segments, palette_dict = process_image(image_name)
            # log_info(f"处理图像{processed_image_name}成功!")
            existing_record = UploadedImage.objects.filter(image_name=image_name).first()

            # 如果找到了现有记录，则更新相关字段
            if existing_record:
                existing_record.processed_image_name = ""
                existing_record.segments = []
                existing_record.palette_dict = {}
                existing_record.state = 0
                existing_record.uploaded_at = datetime.now()
                existing_record.save()
                uploaded_image = existing_record
            else:
                print("开始保存")
                new_record = UploadedImage(image_name=image_name, processed_image_name="",
                                           segments=[], palette_dict={},state=0)
                # UploadedImage.objects.insert([new_record])
                new_record.save()
                print("保存完成")
                uploaded_image = new_record

            jsonstr = json.loads(uploaded_image.to_json())

            id = uploaded_image.id
            jsonstr['id'] = str(id)
            print("返回相应数据")
            return JsonResponse({'status': 'success', 'data': jsonstr, 'message': 'Image uploaded successfully.'})
        except Exception :
            log_error("保存处理前图像出错：",Exception )
            print(Exception )
            return JsonResponse({'status': 'error', 'message': 'save ori_image error'}, status=500)

    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed.'}, status=405)

from seg_app.utils.segResponse import *
# 上传一张图片的原始图片名称进行查询,比如7.jpg
@csrf_exempt
def find_img_byname(request):
    image_name = request.GET.get("image_name")
    img = UploadedImage.objects.filter(image_name=image_name).first()
    log_info(f"查询图片{image_name}")
    if img:
        # jsonstr = json.loads(img.to_json())
        log_info(f"查询成功：{image_name}")
        return jsonRes("success",SUCCESS,img,"find OK")
    else:
        log_info(f"查询失败：{image_name}")
        return jsonRes("warning",WARNING,"","can\'t find this image!")
    pass

from django.utils import timezone
from pytz import timezone as pytz_timezone
# 获取北京时区
beijing_tz = pytz_timezone('Asia/Shanghai')
# 上传时间进行查询
@csrf_exempt
def find_img_byperiod(request):
    time_format = "%Y-%m-%d %H:%M:%S"
    try:
        startDateTime = datetime.strptime(request.GET.get("startDateTime"),time_format).astimezone(beijing_tz)
        endDateTime = datetime.strptime(request.GET.get("endDateTime"),time_format).astimezone(beijing_tz)
    except:
        return jsonRes("error", ERROR,{}, "datetime format error!")
    try:
        images = UploadedImage.objects.filter(uploaded_at__gte=startDateTime, uploaded_at__lte=endDateTime)
    except:
        return jsonRes("error", ERROR, {}, "find images error!")
    return jsonRes("success",SUCCESS,images,"find OK!")

def while_process_imgs():
    last_time = datetime.now()
    seg_img = SegImg(settings.MODEL_CONFIG, settings.CHECKPOINT)
    while True:
        # 间隔3s处理一次图片
        if(datetime.now()-last_time>=timedelta(seconds=3)):
            last_time = datetime.now()
        else:
            continue
        # 读取mongodb数据库
        try:
            img_record = UploadedImage.objects.filter(state=0).order_by("+order_by").first()
            if img_record:
                print("******查询处理")
                log_info(f"******开始处理查询:{img_record.image_name}******")
                new_image_name = idgen.get_id() + '.' + img_record.image_name.split('.')[-1]
                segments, palette_dict = seg_img.prefict(os.path.join(settings.MEDIA_ROOT, img_record.image_name),
                                                         new_image_name)

                img_record.segments = segments
                img_record.palette_dict = palette_dict
                img_record.processed_image_name = new_image_name
                img_record.state = 1
                img_record.save()
                print("******查询处理--结束")
                log_info(f"******结束查询处理:{img_record.image_name}******")
        except Exception as e:
            print("处理图片时出错：", e)
            log_error("处理图片时出错：" + str(e))

def save_image(img,img_name):
    img_path = os.path.join(settings.MEDIA_ROOT, img_name)
    try:
        # 写入文件
        with open(img_path, 'ab') as fp:
            # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
            for chunk in img.chunks():
                fp.write(chunk)
        return 1
    except:
        return 0

# 图片名称需要唯一
# def process_image(image_name):
#     new_image_name = str(idgen.get_id())+'.'+image_name.split('.')[-1]
#     image_path = os.path.join(settings.MEDIA_ROOT, image_name)
#     segments,palette_dict = seg_img.prefict(image_path,new_image_name)
#     print("===处理")
#     return new_image_name,segments,palette_dict
#     pass

@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        video = request.FILES.get("video")
        video_name = request.FILES.get("video").name
        name = request.POST.get("video_name")
    if save_video(video,video_name) ==0:
        return JsonResponse({'status': 'error', 'message': 'save error'}, status=500)


    else:
        return JsonResponse({'status': 'error', 'message': 'Only POST requests are allowed.'}, status=405)

def save_video(vid,vid_name):
    vid_path = os.path.join(settings.MEDIA_ROOT+'\\videos', vid_name)
    try:
        # 写入文件
        with open(vid_path, 'ab') as fp:
            # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
            for chunk in vid.chunks():
                fp.write(chunk)
        return 1
    except:
        return 0
"""
处理图片，需要实时更新处理的进度
@params:
vid_name:视频名称（需要唯一）
如果处理过程中失败了，需要进行反馈和状态填写
"""
def prcess_video(vid_name):
    vid_path = os.path.join(settings.MEDIA_ROOT + '\\videos', vid_name)
    try:
        pass
    except:
        pass
    return "处理视频"+vid_path