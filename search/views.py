from django.shortcuts import render
import cv2
from django.apps import apps
from mtcnn import MTCNN
import psycopg2
from rest_framework.views import APIView
from rest_framework import viewsets
from facenet_pytorch import InceptionResnetV1
from pymilvus import Milvus, DataType, Collection, connections
import numpy as np
from rest_framework.decorators import authentication_classes, permission_classes, action
import pytz
from minio import Minio
import requests
from io import BytesIO
from uuid import uuid4
from rest_framework.parsers import MultiPartParser, FormParser
import torch
from rest_framework.permissions import IsAuthenticated
import base64
from datetime import datetime, timedelta
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed
import insightface
from django.contrib.auth.models import User
from collections import Counter
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import math
from PIL import Image
from metadata.models import Person, SearchHistory, Account, Gallery

detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
milvus = Milvus(host='localhost', port='19530')
connections.connect(
    host='localhost',
    port='19530'
)
rec_model_path = '/root/eTanuReincarnation/metadata/insightface/models/w600k_mbf.onnx'
rec_model_path_windows = ('C:/Users/User4/PycharmProjects/eTanuReincarnationAPI/metadata/insightface/models/w600k_mbf'
                          '.onnx')
rec_model = model_zoo.get_model(rec_model_path_windows)
rec_model.prepare(ctx_id=0)
collection = Collection('face_embeddings')
collection.load()
minio_client = Minio(
    endpoint='127.0.0.1:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False  # Set to True if using HTTPS
)


def search_faces_in_milvus(embedding, limit):
    search_params = {"metric_type": "L2", "params": {"nprobe": 32}}

    results = collection.search(
        anns_field="face_vector",
        data=[embedding],
        limit=limit,
        param=search_params
    )
    # Retrieve vector IDs of the closest embeddings
    vector_ids = [result.id for result in results[0]]
    distances = [result.distance for result in results[0]]

    return vector_ids, distances


def convert_image_to_embeddingv2(img, face):
    # Detect faces in the image
    rec_model.get(img, face)
    embeddings = face.normed_embedding
    return embeddings.squeeze().tolist()


def upload_image_to_minio(image_data, bucket_name, content_type):
    try:
        # Create BytesIO object from image data
        image_stream = BytesIO(image_data)

        # Generate unique object name using uuid4()
        object_name = str(uuid4()) + content_type.replace('image/',
                                                          '.')  # Example: '7f1d18a4-2c0e-47d3-afe1-6d27c3b9392e.png'
        # Upload image to MinIO
        minio_client.put_object(
            bucket_name,
            object_name,
            image_stream,
            len(image_data),
            content_type=content_type  # Change content type based on image format
        )
        return object_name
    except Exception as err:
        print(f"MinIO Error: {err}")


class SearchView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    @action(detail=False, methods=['post'])
    def post(self, request):
        # Get the uploaded image file and the limit parameter from the request

        limit = int(request.POST.get('limit', 5))  # Default limit is 10 if not provided
        user_id = request.POST.get('auth_user_id')
        reload = request.POST.get('reload')
        bucket_name = 'history'
        if reload == "1":
            image_name = request.POST.get('image_name')
            image_url = f'http://127.0.0.1:9000/{bucket_name}/{image_name}'
            response = requests.get(image_url)
            image_data = response.content
        # Read the image file and convert it to an OpenCV format
        else:
            image_file = request.FILES['image']
            image_data = image_file.read()

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({'error': 'Failed to decode the image'}, status=400)

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use MTCNN to detect faces and keypoints in the image
        faces = detector.detect_faces(img_rgb)

        face_results = []
        for face in faces:
            # Convert the face to an embedding

            bbox = face['box']
            det_score = face['confidence']
            kps_dict = face['keypoints']
            kps = np.array([list(kps_dict.values())]).squeeze()

            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            embedding = convert_image_to_embeddingv2(img_rgb, face)
            # Search for the face in Milvus
            vector_ids, distances = search_faces_in_milvus(embedding, limit)
            # Retrieve metadata for each vector ID
            gallery_objects = Gallery.objects.filter(vector_id__in=vector_ids)
            # Prepare data for response

            milvus_results = [{'vector_id': vector_id, 'distance': round(dist, 2)} for vector_id, dist in
                              zip(vector_ids, distances)]
            metadata_list = [
                {'vector_id': obj.vector_id, 'iin': obj.personId.iin, 'name': obj.personId.firstname, 'surname': obj.personId.surname,
                 'patronymic': obj.personId.patronymic, 'birth_date': obj.personId.birthdate, 'photo': obj.photo} for obj in
                gallery_objects]

            # Associate metadata with Milvus results based on vector ID
            for milvus_result in milvus_results:
                vector_id = milvus_result['vector_id']
                metadata = next((item for item in metadata_list if item['vector_id'] == vector_id), None)
                milvus_result['metadata'] = metadata

            keypoints = face.kps
            keypoints = keypoints.tolist()
            bbox = face.bbox

            face_result = {
                'bbox': bbox,
                'keypoints': keypoints,
                'milvus_results': milvus_results
            }
            face_results.append(face_result)

        uploaded_object_name = upload_image_to_minio(image_data, bucket_name, content_type='image/png')
        user = User.objects.get(id=user_id)
        account = Account.objects.get(user=user)

        SearchHistory.objects.create(
            account=account,
            searchedPhoto=uploaded_object_name,
            created_at=datetime.now()
        )

        return JsonResponse({'faces': face_results,
                             'image_name': uploaded_object_name})
