from typing import Union, Optional, Tuple

import cv2
# from mtcnn.mtcnn import MTCNN
import numpy as np
import torch
from PIL import Image
# from facenet_pytorch.models.mtcnn import MTCNN, prewhiten
from facenet_pytorch import MTCNN, prewhiten
from torch.backends import cudnn
from torchvision.transforms import functional

input_image_size = 160
Image_Type = Union[Image.Image, np.ndarray]


class FaceDetector:
    def __init__(self):
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=input_image_size, prewhiten=True, select_largest=True,
            device=self.device
        )
        # self.detector = MTCNN()

    def pre_process(self, image: Image_Type):
        """
        Redimensiona e preprocessa imagem para extracao de features
        :param image: imagem para pre-processamento (cv2 ou Pil)
        :return: img_tensor pre-processado para geracao de embeddings
        """

        image = self.pil_to_cv2(image)

        try:
            image = cv2.resize(image, (input_image_size, input_image_size), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return None
        img_tensor = functional.to_tensor(np.float32(image)).to(self.device)
        return prewhiten(img_tensor)
        # face = F.to_tensor(np.float32(face))

    def detect(self, image: Image_Type) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Realiza deteccao facial e retorna boxes/scores detectados
        :rtype: numpy.ndarray ou None caso nao nenhuma face seja detectada
        :param image: imagem (do Pil ou do cv2) para a deteccao
        :return: arrays boxes com localizacoes das faces e scores, com a probabilidade de presenca de face
        """

        image = self.cv2_to_pil(image)

        boxes, scores = None, None

        try:
            boxes, scores = self.mtcnn.detect(image)
            if boxes is not None:
                boxes = np.rint(boxes).astype(int)

        except Exception as err:
            print(err)

        return boxes, scores

    @staticmethod
    def cv2_to_pil(image: Image_Type) -> Image.Image:
        if type(image) == np.ndarray:
            image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))
        return image

    def extract_face(self, image: Image_Type, save_path: Optional[str] = None):
        """
        Realiza deteccao facial, extrai a imagem da maior face, e pre-processa a imagem para extracao de features
        :rtype: torch.tensor
        :param image: imagem {PIL.Image ou numpy.ndarray do cv2} para a deteccao
        :param save_path: um caminho para salvar a face detectada (opcional)
        :return: imagem da face pre-processada
        """

        def get_img(img: Image_Type = image):
            try:
                img = self.cv2_to_pil(img)

                img_tensor, prob = self.mtcnn(img, save_path=save_path, return_prob=True)
                return img_tensor.to(self.device), prob

            except (RuntimeError, AttributeError) as err:

                return None, None

        res = get_img()
        # if res[0] is None:
        #     for max_dim in [1280, 720, 640, 480, 360]:  # , 3000, 6000]:
        #         # print("Resizing attempt")
        #
        #         res = get_img(self.resize(image, max_dim))
        #         # res = self._extract_without_alignment(self.resize(image, max_dim), save_path)
        #         if res[0] is not None:
        #             print(f"Face encontrada com redimensionamento para {max_dim}")
        #             return res

        return res

    @staticmethod
    def resize(image: Image_Type, max_dim: int, squared: bool = False):
        image = FaceDetector.pil_to_cv2(image)

        old_size = image.shape[:2]
        ratio = max_dim / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))

        if squared:
            delta_w = max_dim - new_size[1]
            delta_h = max_dim - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return image

    @staticmethod
    def pil_to_cv2(image: Image_Type) -> np.ndarray:
        if type(image) == Image.Image:
            # image = Image.fromarray(image)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
