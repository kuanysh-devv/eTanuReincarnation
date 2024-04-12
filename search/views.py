from django.shortcuts import render
import cv2
from mtcnn import MTCNN
import psycopg2
from facenet_pytorch import InceptionResnetV1
from pymilvus import Milvus, DataType, Collection, connections
import numpy as np
import torch
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import math
from PIL import Image
from metadata.models import Metadata

detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
milvus = Milvus(host='localhost', port='19530')
connections.connect(
    host='localhost',
    port='19530'
)
rec_model_path = 'C:/Users/User4/PycharmProjects/eTanuReincarnationAPI/metadata/insightface/models/w600k_mbf.onnx'
rec_model = model_zoo.get_model(rec_model_path)
rec_model.prepare(ctx_id=0)


def search_faces_in_milvus(embedding, limit):
    # Search for the closest embeddings in the Milvus collection
    collection = Collection("face_embeddings")
    collection.load()
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


@csrf_exempt
def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image file and the limit parameter from the request
        image_file = request.FILES['image']
        limit = int(request.POST.get('limit', 10))  # Default limit is 10 if not provided

        # Read the image file and convert it to an OpenCV format
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
            metadata_objects = Metadata.objects.filter(vector_id__in=vector_ids)

            # Prepare data for response

            milvus_results = [{'vector_id': vector_id, 'distance': round(dist, 2)} for vector_id, dist in
                              zip(vector_ids, distances)]
            metadata_list = [{'vector_id': obj.vector_id, 'name': obj.firstname, 'surname': obj.surname,
                              'patronymic': obj.patronymic, 'photo': obj.photo} for obj in metadata_objects]

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

        return JsonResponse({'faces': face_results})

    return JsonResponse({'error': 'Invalid request method or missing image file'}, status=400)
