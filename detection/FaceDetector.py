import time
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np


class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # start_time = time.time()
        result = self.detector.detect_faces(image)
        # elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(elapsed_time))

        # print(f"Deteccao: {result}")

        boxes = []
        confidence = []
        [(boxes.append(i['box']), confidence.append(i['confidence'])) for i in result]
        # keypoints = result[0]['keypoints']

        boxes = np.asarray(boxes)
        confidence = np.asarray(confidence)

        return boxes, confidence

    def extract_face(self, image, min_score=0.5):
        boxes, scores = self.detect(image)
        biggest_area = 0
        cropped_face = None

        # if scores is not None and boxes is not None:
        for i in range(len(scores)):
            if scores[i] > min_score:
                x1, y1, width, height = boxes[i]
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                ith_area = width * height
                if ith_area > biggest_area:
                    cropped_face = image[y1:y2, x1:x2]
                    biggest_area = ith_area
        return cropped_face
