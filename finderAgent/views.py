from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import HttpResponse, JsonResponse
from rest_framework.parsers import JSONParser
from .services import SimFinderModel
from .serializers import SimilaritySerializer
import requests
# Create your views here.


@api_view(['POST'])
def getNumberOfSimilars(request):
    data = JSONParser().parse(request)
    doc = data['sentence']
    numOfsims = int(data['numOfSimilars'])
    simFinder = SimFinderModel()
    simFinder.load()
    similars = simFinder.findSimilars(doc, numOfsims)
    return JsonResponse(SimilaritySerializer(similars, many=True).data, status=201, safe=False)


@api_view(['POST'])
def addDoc(request):
    data = JSONParser().parse(request)
    doc = data['sentence']
    KID = data['KID']

    simFinder = SimFinderModel()
    simFinder.load()

    simFinder.addDoc(doc, KID)

    return JsonResponse(KID, status=201, safe=False)
