from django import forms
from .models import UploadedImage
from PIL import Image

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']

    def __init__(self, *args, **kwargs):
        super(ImageUploadForm, self).__init__(*args, **kwargs)
        self.fields['image'].error_messages = {
            'invalid_image': 'Файл, который вы загрузили, не является изображением или является поврежденным изображением.',
        }

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            try:
                img = Image.open(image)
                if img.format not in ['JPEG', 'PNG', 'JPG', 'MPO']:
                    raise forms.ValidationError('Неподдерживаемый формат изображения.')
            except Exception as e:
                raise forms.ValidationError('Неподдерживаемый формат изображения. Поддерживаются только JPEG, PNG, JPG, MPO')
        return image

class ImageURLForm(forms.Form):
    image_url = forms.URLField(label='URL изображения')
