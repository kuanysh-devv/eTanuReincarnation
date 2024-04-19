from django.db import models
from django.utils.translation import gettext_lazy as _


class Metadata(models.Model):
    vector_id = models.CharField(max_length=50, primary_key=True, verbose_name=_("Vector Id"))
    iin = models.CharField(max_length=255, verbose_name=_("iin"), default="", null=True, blank=True)
    firstname = models.CharField(max_length=255, verbose_name=_("firstName"), default="", null=True, blank=True)
    surname = models.CharField(max_length=255, verbose_name=_("surname"), default="", null=True, blank=True)
    patronymic = models.CharField(max_length=255, verbose_name=_("patronymic"), default="", null=True, blank=True)
    photo = models.TextField(verbose_name=_("Photo"))

    def __str__(self):
        return str(self.surname) + " " + str(self.firstName) + " " + str(self.patronymic)

    class Meta:
        verbose_name = _("Metadata")
        verbose_name_plural = _("Metadata instances")
