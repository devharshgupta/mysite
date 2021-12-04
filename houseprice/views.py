from django.shortcuts import render
from django.http import HttpResponse
from .forms import HomePageForm
# Create your views here.

from .ml import Pridict_price


def index(request):
    context = {}
    form = HomePageForm(request.POST or None)
    context['form'] = form
    if request.POST:
        if form.is_valid():
            bhk = int(form.cleaned_data['bhk'])
            lat = float(form.cleaned_data['lat'])
            long = float(form.cleaned_data['long'])
            area = float(form.cleaned_data['area'])
            resale = int(form.cleaned_data['resale'])
            rera = int(form.cleaned_data['rera'])
            built = int(form.cleaned_data['built'])
            # print(long, lat, bhk, area, resale, rera, built)
            data = Pridict_price(built, rera, bhk, area,
                                 resale, lat, long)
            # print(data[0][0])
            context['price'] = abs(data[0][0])
            return render(request, 'houseprice/index.html', context)

    return render(request, 'houseprice/index.html', context)


def price(request):
    print("************************************* function ran")
    print(request)
    # data = Pridict_price(0, 0, 2, 1022.641509, 1, 26.928785, 75.828002)
    # print("************************************* data printed")
    # print(data)
    return HttpResponse('worked')
    pass
