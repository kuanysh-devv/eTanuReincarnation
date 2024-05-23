from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Person, Account, Gallery
from django.contrib.auth.models import User


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    auth_user_id = serializers.IntegerField(source='user.id', read_only=True)

    def validate(self, attrs):
        data = super().validate(attrs)
        data['auth_user_id'] = self.user.id  # Add auth_user_id to the response data
        return data

    class Meta:
        fields = ('access', 'refresh', 'auth_user_id')  # Include fields needed in the response


class PersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Person
        fields = "__all__"


class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = Account
        fields = "__all__"


class GallerySerializer(serializers.ModelSerializer):
    class Meta:
        model = Gallery
        fields = "__all__"
