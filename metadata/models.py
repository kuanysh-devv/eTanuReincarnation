from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User


class Person(models.Model):
    iin = models.CharField(max_length=255, verbose_name=_("iin"), default="", null=True, blank=True)
    firstname = models.CharField(max_length=255, verbose_name=_("firstName"), default="", null=True, blank=True)
    surname = models.CharField(max_length=255, verbose_name=_("surname"), default="", null=True, blank=True)
    patronymic = models.CharField(max_length=255, verbose_name=_("patronymic"), default="", null=True, blank=True)
    birthdate = models.DateField(verbose_name=_("birthdate"), null=True, blank=True)

    def __str__(self):
        return str(self.surname) + " " + str(self.firstname) + " " + str(self.patronymic)

    class Meta:
        verbose_name = _("Person")
        verbose_name_plural = _("Person instances")


class Gallery(models.Model):
    vector_id = models.CharField(max_length=50, primary_key=True, verbose_name=_("Vector Id"))
    photo = models.TextField(verbose_name=_("Photo"))
    personId = models.ForeignKey(Person, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.personId.surname) + " " + str(self.personId.firstname) + " " + str(self.personId.patronymic) + str(self.photo)

    class Meta:
        verbose_name = _("Gallery")
        verbose_name_plural = _("Gallery instances")


class Account(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='userAccount')
    firstname = models.CharField(max_length=255, verbose_name=_("firstName"), default="", null=True, blank=True)
    surname = models.CharField(max_length=255, verbose_name=_("surname"), default="", null=True, blank=True)
    patronymic = models.CharField(max_length=255, verbose_name=_("patronymic"), default="", null=True, blank=True)
    role_id = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.surname} {self.firstname} {self.patronymic}"

    class Meta:
        verbose_name = _("Account")
        verbose_name_plural = _("Accounts")


class SearchHistory(models.Model):
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='account')
    searchedPhoto = models.TextField(verbose_name=_("Photo"))
    created_at = models.DateTimeField(verbose_name=_("Created at"))

    def __str__(self):
        return f"{self.account.surname} {self.account.firstname} {self.account.patronymic}"

    class Meta:
        verbose_name = _("History")
        verbose_name_plural = _("Histories")
