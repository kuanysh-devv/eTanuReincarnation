from rest_framework import serializers

from .models import Metadata


class MetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Metadata
        fields = "__all__"
