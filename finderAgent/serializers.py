from rest_framework import serializers
from .models import *


class SimilaritySerializer(serializers.ModelSerializer):
    class Meta:
        model = Similarity
        fields = ("KID", "doc", "score")
