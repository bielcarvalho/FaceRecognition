import cv2
# from mtcnn.mtcnn import MTCNN
import numpy as np
import torch
from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN, prewhiten
from torch.backends import cudnn
from torchvision.transforms import functional

input_image_size = 160


class FaceDetector:
    def __init__(self):
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=input_image_size, min_face_size=30, prewhiten=True, select_largest=True,
            device=self.device
        )
        # self.detector = MTCNN()

    def pre_process(self, image):
        """
        Redimensiona e preprocessa imagem para extracao de features
        :param image: imagem do cv2
        :return: img_tensor pre-processado para extracao de features
        """
        try:
            image = cv2.resize(image, (input_image_size, input_image_size), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return None
        img_tensor = functional.to_tensor(np.float32(image)).to(self.device)
        return prewhiten(img_tensor)
        # face = F.to_tensor(np.float32(face))

    def detect(self, image):
        """
        Realiza deteccao facial e retorna boxes/scores detectados
        :rtype: numpy.ndarray ou None caso nao nenhuma face seja detectada
        :param image: imagem (do Pil ou do cv2) para a deteccao
        :return: arrays boxes com localizacoes das faces e scores, com a probabilidade de presenca de face
        """
        if type(image) == np.ndarray:
            image = Image.fromarray(image)

        boxes, scores = self.mtcnn.detect(image)
        if boxes is not None:
            boxes = np.rint(boxes).astype(int)

        return boxes, scores

    def extract_face(self, image, save_path=None):
        """
        Realiza deteccao facial, extrai a imagem da maior face, e pre-processa a imagem para extracao de features
        :rtype: torch.tensor
        :param image: imagem {PIL.Image ou numpy.ndarray do cv2} para a deteccao
        :param save_path: um caminho para salvar a face detectada (opcional)
        :return: imagem da face pre-processada
        """
        if type(image) == np.ndarray:
            image = Image.fromarray(image)

        return self.mtcnn(image, save_path=save_path, return_prob=True)
