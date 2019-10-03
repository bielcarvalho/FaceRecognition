import cv2
import time

import numpy as np
from detection.FaceDetector import FaceDetector
from embeddings.FaceEmbeddings import FaceEmbeddings
from classifier.FaceClassifier import FaceClassifier

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

    # frame = cv2.resize(frame, (750, 750), fx=1, fy=1)  # resize frame (optional)
    curTime = time.time()  # calc fps

    frame = frame[:, :, 0:3]

    boxes, scores = face_detector.detect(frame)
    # face_boxes = boxes[np.argwhere(scores>0.5).reshape(-1)]
    # face_scores = scores[np.argwhere(scores>0.5).reshape(-1)]
    # print('Detected_FaceNum: %d' % len(face_boxes))

    if len(scores) > 0:
        for i in range(len(scores)):
            if scores[i] < 0.5:
                continue
            x1, y1, width, height = boxes[i]
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            cropped_face = frame[y1:y2, x1:x2, :]
            # cv2.imshow(f"Deteccao", cropped_face)
            # cv2.waitKey(0)

            feature = face_recognition.describe(cropped_face)
            name, prob = face_classifier.classify(feature)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # plot result idx under box
            text_x = x1-10
            text_y = y2 + 15

            cv2.putText(frame, f"{name}\n({prob})", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                        thickness=1, lineType=2)
    else:
        print('Unable to align')

    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / sec
    fps_str = 'FPS: %2.3f' % fps
    text_fps_x = len(frame[0]) - 150
    text_fps_y = 20
    cv2.putText(frame, fps_str, (text_fps_x, text_fps_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0),
                thickness=1, lineType=2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

