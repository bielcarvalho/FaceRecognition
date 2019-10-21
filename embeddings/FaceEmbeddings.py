import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class FaceEmbeddings:
    def __init__(self):
        # Load models
        self.embedding_size = 512
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding_size(self):
        return self.embedding_size

    def describe(self, image, to_numpy=True):
        """
        Extrai um array de 512-features da imagem facial recebida
        :rtype: numpy.ndarray ou torch.tensor
        :param image: torch.tensor pre-processado
        :param to_numpy: if True converte o array de features para numpy.ndarray
        :return: array de 512-features
        """
        tensor_arr = self.resnet(image.unsqueeze(0))[0]
        if not to_numpy:
            return tensor_arr
        np_arr = torch.Tensor.cpu(tensor_arr).numpy()
        return np_arr
