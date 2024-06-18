import os
import requests
import uuid
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from .forms import ImageUploadForm, ImageURLForm
from .models import UploadedImage
import subprocess
from PIL import Image, UnidentifiedImageError
from io import BytesIO


RUN_NAME = 'expert'
IMAGE_SIZE = 640
CONF_THRESH = 0.14
WEIGHTS_PATH = 'media/yolov5/runs/train/exp4/weights/best.pt'

def index(request):
    uploaded_image_url = None
    latest_file_url = None
    error_message = None

    form = ImageUploadForm()
    url_form = ImageURLForm()

    if request.method == 'POST':
        if 'upload_image' in request.POST:
            form = ImageUploadForm(request.POST, request.FILES)
            url_form = ImageURLForm()
            try:
                if form.is_valid():
                    uploaded_image_url = handle_file_upload(form)
                else:
                    error_message = form.errors.get('image', ['Ошибка при загрузке файла'])[0]
            except ValidationError as e:
                error_message = str(e)
        elif 'upload_url' in request.POST:
            form = ImageUploadForm()
            url_form = ImageURLForm(request.POST)
            if url_form.is_valid():
                uploaded_image_url, error_message = handle_url_upload(url_form)
        elif 'process_image' in request.POST:
            uploaded_image_url = request.POST.get('uploaded_image_url', None)
            if uploaded_image_url:
                latest_file_url = process_image(uploaded_image_url)
                uploaded_image_url = None
            else:
                error_message = 'Сначала загрузите изображение для обработки.'

    return render(request, 'myapp/index.html', {
        'form': form,
        'url_form': url_form,
        'latest_file': latest_file_url,
        'uploaded_image_url': uploaded_image_url,
        'error_message': error_message,
    })

def handle_file_upload(form):
    uploaded_image = form.save()
    return uploaded_image.image.url

def handle_url_upload(url_form):
    image_url = url_form.cleaned_data['image_url']
    validate = URLValidator()
    try:
        validate(image_url)
    except ValidationError as e:
        return None, 'Неверный URL.'

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return None, 'Время ожидания запроса истекло.'
    except requests.exceptions.RequestException as e:
        return None, f'Ошибка при загрузке изображения: {e}'

    if 'image' not in response.headers.get('Content-Type', ''):
        return None, 'URL не ведет к изображению.'

    try:
        image = Image.open(BytesIO(response.content))
        supported_formats = ['JPEG', 'PNG', 'JPG']
        if image.format not in supported_formats:
            if image.format == 'MPO':
                image = image.convert('RGB')
                image_format = 'JPEG'
            else:
                return None, f'Неподдерживаемый формат изображения. Поддерживаются только {supported_formats}.'
        else:
            image_format = image.format

        image_name = str(uuid.uuid4()) + "_" + image_url.split('/')[-1].rsplit('.', 1)[0] + f".{image_format.lower()}"
        image_content = BytesIO()
        image.save(image_content, format=image_format)
        image_content = ContentFile(image_content.getvalue())

        upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', image_name)
        default_storage.save(upload_path, image_content)
        return settings.MEDIA_URL + 'uploads/' + image_name, None
    except UnidentifiedImageError:
        return None, 'Неподдерживаемый формат изображения.'
    except Exception as e:
        return None, f'Ошибка при обработке изображения: {e}'



def process_image(image_url):
    upload_path = os.path.join(settings.MEDIA_ROOT, image_url.replace(settings.MEDIA_URL, ''))
    command = [
        'python', 'media/yolov5/detect.py',
        '--source', upload_path,
        '--weights', WEIGHTS_PATH,
        '--name', RUN_NAME,
        '--img', str(IMAGE_SIZE),
        '--conf-thres', str(CONF_THRESH),
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:
        return None

    output_dir = os.path.join('media', 'yolov5', 'runs', 'detect', RUN_NAME)
    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    matching_files = [f for f in output_files if upload_path.split('/')[-1] in f]

    if matching_files:
        latest_file = matching_files[0]
        return os.path.join('media', 'yolov5', 'runs', 'detect', RUN_NAME, latest_file)
    return None
