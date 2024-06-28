
from django.db import models as dmodel
# TO USE DJANGO ORM

# BASE ENTİTY
class Base(dmodel.Model):
    id = dmodel.AutoField(primary_key=True,null=False)
    created_at = dmodel.DateTimeField(auto_now_add=True, db_index=True ,null=False)
    updated_at = dmodel.DateTimeField(auto_now=True)
    proxy =  dmodel.BooleanField(null=False, default=False)

    class Meta:
        abstract = True
        app_label= "abmem"
