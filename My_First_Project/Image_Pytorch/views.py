import os

from django.shortcuts import render
from django.http import HttpResponse
from .models import *

current_dir = os.path.dirname(os.path.realpath(__file__))


# Create your views here.
def index(request):
    return render(request, 'index.html', {})


def Sub(request):
    return render(request, 'Sub_File/Sub.html', {})


def handle(request):
    if request.method == 'POST':
        if not os.path.isdir(os.path.join(current_dir, 'uploads')):
            os.mkdir(os.path.join(current_dir, '..', 'uploads'))

        path = os.path.join(current_dir, 'uploads', str(request.FILES['image']))

        # Go to Directory that you named as uploads
        with open(path, 'wb+') as termination:
            for chunk in request.FILES['image'].chunks():
                termination.write(chunk)

        # pred = process([path])
        os.remove(path)
        return render(request, 'Sub_File/handle.html', {'', })
    return HttpResponse("Failed")
