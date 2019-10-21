import time

import cv2
import numpy as np

from classifier.FaceClassifier import FaceClassifier
from detection.FaceDetector import FaceDetector
from embeddings.FaceEmbeddings import FaceEmbeddings

model = input("Choose a model> ")
# model = "knn"

face_detector = FaceDetector()
face_recognition = FaceEmbeddings()
face_classifier = FaceClassifier(f'./classifier/{model.lower()}_classifier.pkl')

video_capture = cv2.VideoCapture(0)
prevTime = 0
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

print('Start Recognition!')
while True:

    ret, frame = video_capture.read()
    # if not ret:
    #     break

    frame = cv2.resize(frame, (0, 0), fx=1, fy=1)  # resize frame (optional)
    curTime = time.time()  # calc fps

    frame = frame[:, :, 0:3]

    boxes, scores = face_detector.detect(frame)

    if boxes is not None and scores is not None:
        boxes = boxes[np.where(scores > 0.6)]
        scores = scores[np.where(scores > 0.6)]
        # detections = face_detector.detect(frame)
        # detections = detections[np.where(detections[:, 4] > 0.5)]
        print('Detected_FaceNum: %d' % len(boxes))

        if len(boxes) > 0:
            for x1, y1, x2, y2 in boxes:
                if x2-x1 < 30 or y2-y1 < 30:
                    continue

                cropped_face = frame[y1:y2, x1:x2, :]
                # cv2.imshow(f"Deteccao", cropped_face)
                # cv2.waitKey(0)

                pre_processed_face = face_detector.pre_process(cropped_face)

                if pre_processed_face is None:
                    continue

                feature = face_recognition.describe(pre_processed_face)
                name, prob = face_classifier.classify(feature)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # plot result idx under box
                text_x = x1-5
                text_y = y2 + 15

                cv2.putText(frame, f"{name} ({prob})", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                            thickness=1, lineType=2)
        else:
            print('Unable to align')

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / sec
    fps_str = 'FPS: %2.3f' % fps
    print(fps_str)
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    cv2.putText(frame, fps_str, (text_fps_x, text_fps_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0),
                thickness=1, lineType=2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

