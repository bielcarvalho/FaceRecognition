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
        :param image: imagem do cv2
        :return: img_tensor pre-processado para extracao de features
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
        # if type(image) == np.ndarray:
        #     image = Image.fromarray(image)

        boxes, scores = None, None

        try:
            boxes, scores = self.mtcnn.detect(image)
            if boxes is not None:
                boxes = np.rint(boxes).astype(int)

        except Exception as err:
            print(err)

        return boxes, scores

    def _extract_without_alignment(self, image: Union[np.ndarray, Image.Image], save_path: Optional[str] = None,
                                   min_score: float = 0.5):
        boxes, scores = self.detect(image)

        if boxes is None or scores is None:
            return None, None

        if len(scores) == 1:
            biggest_pic_idx = 0
        else:
            boxes = boxes[np.where(scores >= min_score)]
            scores: np.ndarray = scores[np.where(scores >= min_score)]

            if scores.size < 1:
                return None, None

            # score = res[:, 4]
            w = boxes[:, 2] - boxes[:, 0] + 1
            h = boxes[:, 3] - boxes[:, 1] + 1
            # best_score_idx = np.argmax(scores, axis=4)
            biggest_pic_idx = np.argmax(np.multiply(w, h))

        x1, y1, x2, y2 = boxes[biggest_pic_idx]
        score = scores[biggest_pic_idx]

        if x2 - x1 < 30 or y2 - y1 < 30:
            return None, None

        if type(image) != np.ndarray:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image = image[y1:y2, x1:x2, :]

        if save_path is not None:
            from os import makedirs, path

            makedirs(path.dirname(save_path), exist_ok=True)

            cv2.imwrite(save_path, image)

        image = self.pre_process(image)

        return image, score

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
                # Ocorre para  dimensao da imagem
                return None, None

        res = get_img()
        if res[0] is None:
            for max_dim in [1280, 720, 640, 480, 360]:  # , 3000, 6000]:
                # print("Resizing attempt")

                res = get_img(self.resize(image, max_dim))
                # res = self._extract_without_alignment(self.resize(image, max_dim), save_path)
                if res[0] is not None:
                    return res

        return res

    @staticmethod
    def resize(image: Image_Type, max_dim: int, squared: bool = False):

        image = FaceDetector.pil_to_cv2(image)

        # old_size = image.size  # old_size[0] is in (width, height) format
        old_size = image.shape[:2]
        # ratio = float(max_dim) / max(old_size)
        ratio = max_dim / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # image.thumbnail(new_size, Image.ANTIALIAS)
        # image = image.resize(new_size, Image.ANTIALIAS)

        image = cv2.resize(image, (new_size[1], new_size[0]))

        if squared:
            # new_im = Image.new("RGB", (max_dim, max_dim))
            # new_im.paste(image, ((max_dim - new_size[0]) // 2, (max_dim - new_size[1]) // 2))
            # image = new_im

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
