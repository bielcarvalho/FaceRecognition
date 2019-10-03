from os import path
import cv2
import numpy as np
import tensorflow as tf
from embeddings import facenet

# BASE_DIR = os. + '/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_PATH = path.join(path.dirname(__file__), "model", "20180402-114759.pb")
input_image_size = 160


class FaceEmbeddings:
    def __init__(self, res_type=0):
        # Load models
        self.embedding_size = 512
        self.resize_type = res_type
        self.recognition_graph = tf.Graph()
        self.sess = tf.Session(graph=self.recognition_graph)
        print('Loading feature extraction model')
        with self.sess.as_default():
            with self.recognition_graph.as_default():
                facenet.load_model(MODEL_PATH)

    def __del__(self):
        self.sess.close()

    def get_embedding_size(self):
        return self.embedding_size

    def resize(self, image, fill_color=[0, 0, 0]):
        global input_image_size
        # Retorna "image" redimensionada, de acordo com as variaveis "resize_type"

        height, width = image.shape[:2]
        new_image = image

        if width != height and self.resize_type < 2:
            # resize_type 0 para cortar sobras, 1 para preencher com preto, e 2 para redimensionar mudando ratio
            if self.resize_type == 0:
                new_size = min(width, height)
            else:
                new_size = max(width, height)

            delta_w = new_size - width
            delta_h = new_size - height
            top, bottom = max(delta_h // 2, 0), delta_h - (delta_h // 2)
            left, right = max(delta_w // 2, 0), delta_w - (delta_w // 2)

            print(top, bottom, left, right)

            if self.resize_type == 0:
                new_image = image[top:new_size-bottom, left:new_size-right]
            else:
                new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)

        # interpolation method
        new_size, _ = new_image.shape[:2]
        if new_size < input_image_size: # stretching image
            interp = cv2.INTER_CUBIC

        else:  # shrinking image
            interp = cv2.INTER_AREA

        return cv2.resize(new_image, (input_image_size, input_image_size), interpolation=interp)

    def describe(self, image):
        global input_image_size
        images_placeholder = self.recognition_graph.get_tensor_by_name("input:0")
        embeddings = self.recognition_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.recognition_graph.get_tensor_by_name("phase_train:0")
        self.embedding_size = embeddings.get_shape()[1]

        emb_array = np.zeros((1, self.embedding_size))
        image = facenet.prewhiten(image)
        image = cv2.resize(image, (input_image_size, input_image_size), interpolation=cv2.INTER_AREA)
        # image = self.resize(image)
        # cv2.imshow(f"{self.resize_type}", image)
        # cv2.waitKey(0)

        image = image.reshape(-1, input_image_size, input_image_size, 3)
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        emb_array[0, :] = self.sess.run(embeddings, feed_dict=feed_dict)
        return emb_array.squeeze()
