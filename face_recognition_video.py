import argparse
import time

import cv2
import numpy as np

from classifier.face_classifier import FaceClassifier
from detection.face_detector import FaceDetector
from embeddings.face_embeddings import FaceEmbeddings
from face_recognition_train import models


def main():
    ap = argparse.ArgumentParser()
    try:
        models.remove("all")
    finally:
        ap.add_argument("-clf", "--classifier", default="svm", const="svm", nargs='?', choices=models,
                        help=f"classificador responsavel pelo reconhecimento facial ({models})")
        args = vars(ap.parse_args())

    face_detector = FaceDetector()
    face_recognition = FaceEmbeddings()
    face_classifier = FaceClassifier(model_name=args["classifier"])

    video_capture = cv2.VideoCapture(0)
    prev_time = 0
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    print('Iniciando reconhecimento!')
    while True:

        ret, frame = video_capture.read()
        # if not ret:
        #     break

        frame = cv2.resize(frame, (0, 0), fx=1, fy=1)  # resize frame (optional)
        curr_time = time.time()  # calc fps

        frame = frame[:, :, 0:3]

        boxes, scores = face_detector.detect(frame)

        if boxes is not None and scores is not None:
            boxes = boxes[np.where(scores > 0.6)]
            scores = scores[np.where(scores > 0.6)]
            # detections = face_detector.detect(frame)
            # detections = detections[np.where(detections[:, 4] > 0.5)]
            print('Faces detectadas: %d' % len(boxes))

            if len(boxes) > 0:
                for x1, y1, x2, y2 in boxes:
                    if x2 - x1 < 30 or y2 - y1 < 30:
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
                    text_x = x1 - 5
                    text_y = y2 + 15

                    def legenda(text: str):
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255), thickness=1, lineType=2)

                    if args["classifier"] == "svm":
                        legenda(f"{name} ({prob})") if prob >= 0.25 else legenda(f"Desconhecido")

                    elif args["classifier"] == "mlp":
                        legenda(f"{name} ({prob})") if prob >= 0.95 else legenda(f"Desconhecido")

                    else:
                        legenda(f"{name} ({prob})")

            else:
                print('Unable to align')

        sec = curr_time - prev_time
        prev_time = curr_time
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


if __name__ == "__main__":
    main()
