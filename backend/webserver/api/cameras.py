from flask import request
from flask_restplus import Namespace, Resource, reqparse, inputs
from flask_login import login_required, current_user
from werkzeug.datastructures import FileStorage
from mongoengine.errors import NotUniqueError
from mongoengine.queryset.visitor import Q
from threading import Thread
from flask import send_file

from google_images_download import google_images_download as gid

from ..util.pagination_util import Pagination
from ..util import query_util, coco_util, profile

from database import (
    ImageModel,
    DatasetModel,
    CategoryModel,
    AnnotationModel,
    ExportModel
)

from PIL import Image
import datetime
import json
import os
import io

from ..cache import cache

import logging
logger = logging.getLogger('gunicorn.error')

api = Namespace('dataset', description='Dataset related operations')


@api.route('/')
class Cameras(Resource):

    def get(self):

        return query_util.fix_ids(DatasetModel.objects(deleted=False).only('display_name', 'country', 'province', 'city', 'latitude', 'longitude').all())
        #return send_file('/datasets/aggregations.csv', mimetype='text/csv', attachment_filename='pLitterCCTVs.csv', as_attachment=True)

    def get(self, country):

        if country:
            return query_util.fix_ids(DatasetModel.objects(deleted=False, country=country).only('display_name', 'country', 'province', 'city', 'latitude', 'longitude').all())

    def get(self, country, province):

        if country and province:
            return query_util.fix_ids(DatasetModel.objects(deleted=False, country=country, province=province).only('display_name', 'country', 'province', 'city', 'latitude', 'longitude').all())

    def get(self, country, province, city):

        if country and province and city:
            return query_util.fix_ids(DatasetModel.objects(deleted=False, country=country, province=province, city=city).only('display_name', 'country', 'province', 'city', 'latitude', 'longitude').all())
