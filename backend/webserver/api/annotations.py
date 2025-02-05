
from flask_restplus import Namespace, Resource, reqparse
from flask_login import login_required, current_user

from database import (
    ImageModel,
    AnnotationModel
)
from ..util import query_util

import datetime
import logging
logger = logging.getLogger('gunicorn.error')

api = Namespace('annotation', description='Annotation related operations')

create_annotation = reqparse.RequestParser()
create_annotation.add_argument(
    'image_id', type=int, required=True, location='json')
create_annotation.add_argument('category_id', type=int, location='json')
create_annotation.add_argument('isbbox', type=bool, location='json')
create_annotation.add_argument('metadata', type=dict, location='json')
create_annotation.add_argument('segmentation', type=list, location='json')
create_annotation.add_argument('bbox', type=list, location='json')
create_annotation.add_argument('keypoints', type=list, location='json')
create_annotation.add_argument('color', location='json')

update_annotation = reqparse.RequestParser()
update_annotation.add_argument('category_id', type=int, location='json')
update_annotation.add_argument('bbox', type=list, location='json', default=[])
update_annotation.add_argument('segmentation', type=list, location='json', default=[])

@api.route('/')
class Annotation(Resource):

    @login_required
    def get(self):
        """ Returns all annotations """
        return query_util.fix_ids(current_user.annotations.exclude("paper_object").all())

    @api.expect(create_annotation)
    # @login_required
    # add condition if dataset is public
    def post(self):
        """ Creates an annotation """
        args = create_annotation.parse_args()
        image_id = args.get('image_id')
        category_id = args.get('category_id')
        isbbox = args.get('isbbox')
        metadata = args.get('metadata', {})
        segmentation = args.get('segmentation', [])
        bbox = args.get('bbox', [])
        keypoints = args.get('keypoints', [])   

        # change appro after setting login_user
        # image = current_user.images.filter(id=image_id, deleted=False).first()
        # image = ImageModel.objects.get(id=image_id, deleted=False).first()
        image = ImageModel.objects(id=image_id).first()
        if image is None:
            return {"message": "Invalid image id"}, 400
        
        logger.info(
            f'{current_user.username} has created an annotation for image {image_id} with {isbbox}')
        logger.info(
            f'{current_user.username} has created an annotation for image {image_id}')


        # add condition if user is not authenticed or dataset is public
        try:
            annotation = AnnotationModel(
                image_id=image_id,
                category_id=category_id,
                metadata=metadata,
                segmentation=segmentation,
                bbox=bbox,
                keypoints=keypoints,
                isbbox=isbbox
            )
            annotation.save()
        except (ValueError, TypeError) as e:
            return {'message': str(e)}, 400

        return query_util.fix_ids(annotation)


@api.route('/<int:annotation_id>')
class AnnotationId(Resource):

    @login_required
    def get(self, annotation_id):
        """ Returns annotation by ID """
        annotation = current_user.annotations.filter(id=annotation_id).first()

        if annotation is None:
            return {"message": "Invalid annotation id"}, 400

        return query_util.fix_ids(annotation)

    @login_required
    def delete(self, annotation_id):
        """ Deletes an annotation by ID """
        annotation = current_user.annotations.filter(id=annotation_id).first()

        if annotation is None:
            return {"message": "Invalid annotation id"}, 400
        logger.info(f'{annotation.id, annotation.image_id}')
        if annotation.image_id:
            image = current_user.images.filter(id=annotation.image_id, deleted=False).first()
            image.flag_thumbnail()

        annotation.update(set__deleted=True,
                          set__deleted_date=datetime.datetime.now())
        return {'success': True}

    @api.expect(update_annotation)
    # @login_required
    def put(self, annotation_id):
        """ Updates an annotation by ID """
        annotation = current_user.annotations.filter(id=annotation_id).first()

        if annotation is None:
            return { "message": "Invalid annotation id" }, 400

        args = update_annotation.parse_args()

        new_category_id = args.get('category_id')
        new_bbox = args.get('bbox', [])   
        new_segmentation = args.get('segmentation', [])
 
        if len(new_bbox) == 0:
            annotation.update(category_id=new_category_id)
            logger.info(
                f'{current_user.username} has updated category for annotation (id: {annotation.id})'
            )
            newAnnotation = current_user.annotations.filter(id=annotation_id).first()
            return query_util.fix_ids(newAnnotation)
        else:
            annotation.update(category_id=new_category_id, bbox=new_bbox, segmentation=new_segmentation, paper_object=[])
            logger.info(
                f'{current_user.username} has updated bbox for annotation (id: {annotation.id})'
            )
            newAnnotation = current_user.annotations.filter(id=annotation_id).first()
            return query_util.fix_ids(newAnnotation)


# @api.route('/<int:annotation_id>/mask')
# class AnnotationMask(Resource):
#     def get(self, annotation_id):
#         """ Returns the binary mask of an annotation """
#         return query_util.fix_ids(AnnotationModel.objects(id=annotation_id).first())

@api.route('/<int:image_id>/predictions')
class ImageId(Resource):

    @login_required
    def post(self):
        return {}


