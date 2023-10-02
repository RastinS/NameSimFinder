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
    print ('Hello, saeed! ')
    data = JSONParser().parse(request)
    doc = data['sentence']
    print (doc)
    numOfsims = int(data['numOfSimilars'])
    print (numOfsims)
    simFinder = SimFinderModel()
    print ('111111111')
    simFinder.load()
    print ('22222222')
    similars = simFinder.findSimilars(doc, numOfsims)
    print ('3333333')
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
