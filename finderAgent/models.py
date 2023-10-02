from django.db import models

# Create your models here.


class Similarity(models.Model):
    KID = models.CharField(max_length=200)
    doc = models.CharField(max_length=2000)
    score = models.FloatField()

    class Meta:
        managed = False
