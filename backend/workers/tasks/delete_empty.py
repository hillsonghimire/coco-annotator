
from database import (
    ImageModel,
    AnnotationModel,
    TaskModel,
    DatasetModel
)

from celery import shared_task
from ..socket import create_socket
from .thumbnails import thumbnail_generate_single_image

import os
import datetime


@shared_task
#@task
def delete_empty_images_in_dataset(task_id, dataset_id, start_date, end_date):

    task = TaskModel.objects.get(id=task_id)
    dataset = DatasetModel.objects.get(id=dataset_id)

    task.update(status="PROGRESS")
    socket = create_socket()

    directory = dataset.directory
    images_prefix = dataset.images_prefix if dataset.images_prefix else ''
    start_date = images_prefix+start_date
    end_date = images_prefix+end_date

    # toplevel = sorted([im for im in os.listdir(directory) if (im > start_date and im < end_date)])
    task.info(f"Scanning {dataset.name}")

    db_images = ImageModel.objects(dataset_id=dataset.id, file_name__gte=str(start_date), file_name__lte=str(end_date)).all()

    task.info(f"Found images: {db_images.count()}")

    count = 0
    youarehere = 0

    for db_image in db_images:
        progress = int(((youarehere)/db_images.count())*100)
        task.set_progress(progress, socket=socket)
        file_name = db_image.file_name
        task.info(f'{file_name}')
        path = os.path.join(dataset.directory, file_name)
        thumbnail_path = os.path.join(dataset.directory, db_image.thumbnail_path())

        image_id = db_image.id
        image_annotations = AnnotationModel.objects(image_id=image_id).all()
        is_predicted = False
        added_categories = set()
        instance_count = {}
        for ann in image_annotations:
            added_categories.add(ann.category_id)
            if ann.category_id in instance_count.keys():
                instance_count[str(ann.category_id)] += 1
            else:
                instance_count[str(ann.category_id)] = 1
        db_image.update(
            set__num_annotations=len(image_annotations),
            set__annotated=(len(image_annotations) > 0),
            set__category_ids=list(added_categories),
            set__instances=instance_count,
            set__regenerate_thumbnail=False,
            set__is_predicted_with=True
        )
        task.info(f'{instance_count}')
        if image_annotations.count() == 0:
            db_image.update(set__deleted=True, set__deleted_date=datetime.datetime.now())
            task.info(f"Deleting empty image {path}")
            if os.path.isfile(path):
                os.remove(path)
                os.remove(thumbnail_path)
            count += 1
        youarehere += 1

    task.info(f"Deleted {count} empty image(s) from {youarehere} images.")
    task.set_progress(100, socket=socket)


__all__ = ["delete_empty_images_in_dataset"]
