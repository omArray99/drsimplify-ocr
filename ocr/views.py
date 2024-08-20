# # ocr/views.py
# from django.views.decorators.csrf import csrf_exempt
# from django.http import JsonResponse
# from django.shortcuts import render
# from .forms import ImageUploadForm
# from .models import UploadedImage
# from .ocr_utils import infer  # Correct import statement

# def main(request):
#     return render(request, 'ocr/upload.html')


# @csrf_exempt
# def upload_image(request):
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded_image = form.save()
#             result = infer(uploaded_image.image.path)
#             return JsonResponse({'result': result})
#         else:
#             return JsonResponse({'error': 'Invalid form submission'}, status=400)
#     return JsonResponse({'error': 'Invalid request method'}, status=405)

# # def index(request):
# #     return render(request, 'ocr/upload.html', {'form': ImageUploadForm()})



from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from .ocr_utils import infer  # Correct import statement

def main(request):
    return render(request, 'ocr/upload.html')

@csrf_protect
def upload_image(request):
    if request.method == 'POST':
        print("Request POST:", request.POST)
        print("Request FILES:", request.FILES)
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            result = infer(uploaded_image.image.path)
            return JsonResponse({'result': result})
        else:
            print("Form errors:", form.errors)
            return JsonResponse({'error': 'Invalid form submission'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
