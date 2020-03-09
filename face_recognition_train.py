import argparse
import sys
from os import path, makedirs, scandir
from typing import Optional, Tuple, Union, List, Iterable, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from classifier.face_classifier import FaceClassifier

models = [
    "knn",
    "svm",
    "mlp",
    "all"
]

# save_images = False
proj_folder = path.dirname(__file__)
input_folder = path.join(proj_folder, "data", "input")
output_folder = path.join(proj_folder, "data", "output")

# Salvar dataframe embeddings e people em csvs
csv_dfs_output = False
VECTOR_SIZE = 512


def download(file_path, url):
    """
    Realiza download de uma url em uma localizacao desejada
    :param file_path: caminho para salvar o conteudo baixado
    :param url: fonte do download
    """
    import requests
    import math

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    block_size = 1024

    with open(file_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), desc="Download",
                         unit='B', unit_scale=True, unit_divisor=1024):
            f.write(data)


def df_tolist(df: pd.DataFrame) -> list:
    """
    Converte um DataFrame para uma lista, mantendo indices e valores
    :param df: DataFrame a ser convertido
    :return: lista representando df
    """
    return [[index] + value for index, value in zip(df.index.tolist(), df.values.tolist())]


class TrainTestRecognition:

    def __init__(self, rand_seed: int = 42, download_images: bool = False, append: bool = False,
                 save_images: bool = False):
        """

        :param rand_seed: seed usada na geracao de numeros pseudo aleatorios
        :param download_images: baixar banco de imagens LFW
        :param append: adicionar as imagens da atual pasta de entrada aos dfs ja salvos
        :param save_images: salvar imagens das faces recortadas
        """
        self.people_folders: Optional[List[str]] = None
        self.number_imgs_list: List[List[str, int, int]] = []
        self.embeddings: List[Iterable] = []
        self.embeddings_ids: list = []

        self.embeddings_df: Optional[pd.DataFrame] = None
        self.people_df: Optional[pd.DataFrame] = None
        self.images_per_person: Optional[int] = None
        self.save_images = save_images

        if rand_seed >= 0:
            self.random_seed = rand_seed
            print(f"Reproducibilidade possivel com seed {self.random_seed}")
        else:
            self.random_seed: Optional[int] = None

        if download_images is True:
            self._down_img_db()

        self.classifiers: List[str] = []

        if not self._load_dfs() or append:
            assert path.exists(input_folder), f"A pasta de entrada {input_folder} nao existe, informe uma " \
                                              f"pasta com imagens para deteccao, ou utilize a opcao '-down' " \
                                              f"para baixar imagens do banco LFW"
            self._detect_faces(append)

    def train(self, classifier: str, tune_parameters: bool = True, num_sets: int = 5, images_per_person: int = 10,
              optimize_number_images: bool = False, num_images_test: int = 5, num_images_train: int = 10):
        """
        Executa treinamentos e testes
        :param classifier: nome do classificador a ser utilizado ('all' para todos)
        :param tune_parameters: se deve ser executada otimizacao dos hiperparametros utilizando bayes
        :param num_sets: numero de sets para k-fold (usado junto com tune_parameters)
        :param images_per_person: numero de imagens por pessoa a ser dividida entre os sets (usado junto com
        tune_parameters)
        :param optimize_number_images: se devem ser executados testes de otimizacao de numero minimo de imagens por
        pessoa
        :param num_images_test: numero de imagens para utilizar em teste (utilizado quando tune_parameters == False)
        :param num_images_train: numero de imagens para utilizar em treinamento (utilizado quando
        tune_parameters == False)
        """
        global models

        clf_name = classifier.lower()
        if clf_name == "all":
            self.classifiers = models.copy()
            self.classifiers.remove("all")
        else:
            self.classifiers.append(clf_name)

        if tune_parameters:
            self.images_per_person = images_per_person
            assert num_sets > 1, f"Para realizar cross-validation, precisa haver sets de treinamento e testes " \
                                 f"(self.num_sets >= 2)"
            assert (self.images_per_person >= num_sets) and \
                   (self.images_per_person % num_sets == 0), \
                f"Deve haver ao menos uma imagem por set para o cross-validation, e o valor deve ser proporcional" \
                f" ao numero de sets para que as diferentes classes tenham a mesma probabilidade de serem " \
                f"classificadas"

        else:
            self.images_per_person = num_images_test + num_images_train
            num_sets = num_images_train / self.images_per_person

        if optimize_number_images:
            self._optimize_num_images(num_images_train, num_images_test)

        else:
            shuffled_idx = self._get_shuffled_indices()

            print(f"\nSelecionando {len(shuffled_idx) * self.images_per_person} imagens "
                  f"de {len(shuffled_idx)} pessoas com mais de {self.images_per_person} imagens")

            X, Y, shuffled_idx, images_test_ids = self._select_embeddings(self.images_per_person, shuffled_idx)
            num_people = len(shuffled_idx)

            face_classifier = FaceClassifier(self.random_seed, tune_parameters)

            for model in self.classifiers:
                print(f"Treinando modelo {model}")
                face_classifier.train(X, Y, model_name=model, num_sets=num_sets,
                                      images_per_person=(self.images_per_person if tune_parameters
                                                         else (num_images_train, num_images_test)),
                                      num_people=num_people,
                                      test_images_id=images_test_ids)

    @staticmethod
    def _down_img_db():
        """
        Baixa conjunto de imagens do banco de imagens LFW na pasta de entrada informada
        """
        import shutil
        import tarfile

        input_parent_dir = path.dirname(input_folder)
        temp_folder = path.join(path.curdir, "data", "temp")

        makedirs(input_parent_dir, exist_ok=True)
        makedirs(temp_folder, exist_ok=True)

        if path.exists(input_folder):
            shutil.move(input_folder, input_folder + "_bkp")

        tgz_path = path.join(temp_folder, "lfw.tgz")
        download(tgz_path, "http://vis-www.cs.umass.edu/lfw/lfw.tgz")

        if not path.exists(tgz_path):
            print("Problema no download")
            sys.exit()

        print("Extraindo arquivo para {}, isso pode levar um tempo".format(temp_folder))

        tar = tarfile.open(tgz_path, "r:gz")
        tar.extractall(temp_folder)

        print("Movendo arquivos extraidos para a pasta de entrada")
        shutil.move(path.join(temp_folder, "lfw"), input_folder)

        return True

    def _embeddings_to_df(self):
        """
        Salva a lista de embeddings de imagens no dataframe 'self.embeddings_df', e no arquivo compactado
        'embeddings.bz2' para uso posterior
        """
        global VECTOR_SIZE, output_folder, csv_dfs_output
        index = pd.MultiIndex.from_tuples(self.embeddings_ids, names=["Name", "Image_Number"])

        temp = pd.DataFrame(self.embeddings,
                            columns=[("v" + str(i)) for i in range(VECTOR_SIZE)],
                            index=index)

        if self.embeddings_df is not None:
            self.embeddings_df = self.embeddings_df.append(temp)
        else:
            self.embeddings_df = temp

        self.embeddings_df.to_pickle(path.join(output_folder, "embeddings.bz2"))
        if csv_dfs_output:
            self.embeddings_df.to_csv(path.join(output_folder, "embeddings.csv"), sep=";")

        del self.embeddings, self.embeddings_ids

    def _people_to_df(self):
        """
        Salva a lista das pessoas no dataframe 'self.people_df', e no arquivo compactado 'people.bz2' para uso posterior
        """
        global output_folder, csv_dfs_output
        temp = pd.DataFrame(self.number_imgs_list, columns=["Name", "Number_Images", "Not_Found"])
        temp.set_index("Name", inplace=True)

        if self.people_df is not None:
            self.people_df = self.people_df.append(temp)
        else:
            self.people_df = temp

        self.people_df.to_pickle(path.join(output_folder, "people.bz2"))
        if csv_dfs_output:
            self.people_df.to_csv(path.join(output_folder, "people.csv"), sep=";")

        del self.number_imgs_list

    def _load_dfs(self) -> bool:
        """
        Carrega dataframes de pessoas identificadas e os embeddings de suas imagens
        :return: True se conseguir carregar, ou false, se nao conseguir
        """
        global output_folder, input_folder
        people_file = path.join(output_folder, "people.bz2")
        embeddings_file = path.join(output_folder, "embeddings.bz2")
        if not path.exists(people_file) and not path.exists(embeddings_file):
            return False

        self.people_df = pd.read_pickle(people_file).infer_objects()
        self.embeddings_df = pd.read_pickle(embeddings_file).infer_objects()

        return True

    def _detect_faces(self, append: bool = False):
        """
        Percorre pastas de pessoas na pasta de entrada, detectando faces nas imagens e gerando embeddings
        """
        global VECTOR_SIZE, output_folder, input_folder

        from embeddings.face_embeddings import FaceEmbeddings
        from detection.face_detector import FaceDetector
        import cv2

        face_detector = FaceDetector()
        face_recognition = FaceEmbeddings()

        print("\nExecutando deteccao facial")

        if self.people_folders is None:
            self.people_folders = [f.path for f in scandir(input_folder) if f.is_dir()]
        assert len(self.people_folders) >= 1

        prog_bar = tqdm(total=len(self.people_folders), desc="Detectando", position=0, unit="pessoas")

        for person_path in self.people_folders:

            person_name = path.basename(person_path)
            person_imgs_path = [f.path for f in scandir(person_path) if f.is_file()]

            start = 0
            failed_to_detect = 0

            if append:
                try:
                    # Para adicionar novas imagens a dataframes ja formados
                    df_row = self.people_df.loc[person_name]
                    start = int(df_row["Number_Images"])
                    failed_to_detect = int(df_row["Not_Found"])
                    self.people_df.drop(person_name, inplace=True)
                except (KeyError, TypeError, AttributeError) as err:
                    pass

            self.number_imgs_list.append([person_name, len(person_imgs_path) + start, failed_to_detect])

            for i, img_path in enumerate(person_imgs_path, start=start):

                # face_img = path.join(person_path, "MTCNN", f"{str(i)}.jpg")
                # if path.exists(face_img):
                #     import cv2
                #     img = cv2.imread(face_img)
                #
                #     img = face_detector.pre_process(img)
                #
                #     self.embeddings.append(face_recognition.describe(img))
                #     self.embeddings_ids.append([str(person_name), i])
                #
                #     continue

                try:
                    # img = Image.open(img_path)
                    img = cv2.imread(img_path)

                except (OSError, IOError):
                    tqdm.write('Open image file failed: ' + img_path)
                    self.number_imgs_list[-1][-1] += 1
                    continue

                if img is None:
                    tqdm.write('Open image file failed: ' + img_path)
                    self.number_imgs_list[-1][-1] += 1
                    continue

                tqdm.write(f'Detecting image {i}, file: {img_path}')

                # image_torch, score = face_detector.extract_face(img)
                image_torch, score = face_detector.extract_face(img, save_path=(path.join(person_path, "MTCNN",
                                                                                          f"{str(i)}.jpg")
                                                                                if self.save_images is True else None))

                if image_torch is None or score < 0.5:
                    tqdm.write(f'No face found in {img_path}')
                    if score is not None:
                        tqdm.write(f'(Score: {score})')
                    self.number_imgs_list[-1][-1] += 1
                    continue

                self.embeddings.append(face_recognition.describe(image_torch))
                self.embeddings_ids.append([str(person_name), i])
            prog_bar.update(1)
        prog_bar.close()

        VECTOR_SIZE = face_recognition.get_embedding_size()
        makedirs(output_folder, exist_ok=True)
        self._embeddings_to_df()
        self._people_to_df()

        del self.people_folders

    def _get_embeddings_vector(self, person_name: str, img_number: int) -> Optional[np.ndarray]:
        """
        Obtem embedding da pessoa pessoa e imagem desejadas
        :param person_name: nome da pessoa a ser buscada
        :param img_number: numero da imagem desejada da pessoa
        :return: array com vetor de embeddings, ou None caso nao tenha sido localizado
        """
        try:
            return self.embeddings_df.loc[(person_name, img_number)].values
        except (KeyError, TypeError) as kerr:
            # tqdm.write(kerr)
            # tqdm.write(f"ID desejado: {person_name}; Img: {img_number}")
            return None

    def _get_shuffled_indices(self) -> Dict[str, List[int]]:
        """
        Seleciona indices aleatorios de imagens para cada pessoa que contenha ao menos 'self.images_per_person' imagens
        :return: Dictionary com pessoas selecionadas e lista de indices das imagens embaralhados
        """
        people_list = df_tolist(self.people_df.loc[(self.people_df["Number_Images"] - self.people_df["Not_Found"])
                                                   >= self.images_per_person])
        assert len(people_list) > 0, "Nao ha pessoas com a quantidade de imagens desejada"

        shuffled_idxs = {}

        import random

        for person, num_images, not_found in people_list:

            shuffled_idxs[person] = list(range(num_images))

            if self.random_seed is not None:
                random.seed(self.random_seed)
            random.shuffle(shuffled_idxs[person])

        # print(f"Indices de {people_list[0][0]}: {shuffled_idxs[people_list[0][0]]}")

        return shuffled_idxs

    def _optimize_num_images(self, num_images_test: int = 5, num_images_train: int = 10):
        """
        Realiza treinamentos e testes para verificar numero de imagens ideal a ser utilizado por pessoa,
        executando desde 1 imagem para treinamento por pessoa, ate atingir num_images_train.
        :param num_images_test: numero de imagens que sera fixado para teste
        :param num_images_train: numero de imagens limite para testar em treinamento
        """

        try:
            shuffled_idxs = self._get_shuffled_indices()
        except AssertionError:
            raise Exception("Nao ha pessoas com a quantidade de imagens necessaria para efetuar o teste")

        X_test, y_test, shuffled_idxs, test_images_id = self._select_embeddings(num_images_test, shuffled_idxs)
        X_train, y_train, temp = [], [], []

        face_classifier = FaceClassifier(self.random_seed)


        progress_bar = tqdm(total=num_images_train, desc="Otimizando Numero de Imagens",
                            unit="iteracoes", file=sys.stdout,
                            dynamic_ncols=True)

        for i in range(1, num_images_train + 1):
            X_train_new, y_train_new, shuffled_idxs, _ = self._select_embeddings(1, shuffled_idxs)
            X_train.extend(X_train_new)
            y_train.extend(y_train_new)

            best_score, best_model = 0.0, None

            for model in self.classifiers:
                new_score = face_classifier.train(X=X_train, y=y_train, X_test=X_test, y_test=y_test, model_name=model,
                                                  num_sets=i / (i + num_images_test),
                                                  images_per_person=(i, num_images_test),
                                                  num_people=len(shuffled_idxs),
                                                  test_images_id=test_images_id)

                if new_score > best_score:
                    best_score, best_model = new_score, model

            progress_bar.write(f"Melhor com {i} imagens - {best_model}: {best_score}")
            progress_bar.update()

    def _select_embeddings(self, number_images_select: int,
                           people: Dict[str, List[int]]) -> Tuple[Union[list, np.ndarray], list, dict, list]:
        """
        Seleciona embeddings de imagens de acordo com indices de imagens fornecidos por pessoa, e a quantidade desejada.
        :param number_images_select: Numero de embeddings a ser selecionado para cada pessoa
        :param people: Dictionary que relaciona nomes de pessoas e indices de images embaralhados, sendo selecionados
        embeddings das imagens com indices no final da lista, ate atingir o valor desejado (number_images_select)
        :return: X (embeddings), y (labels), dict com indices restantes, e indices das imagens que geraram os embeddings
        """

        global VECTOR_SIZE
        if VECTOR_SIZE is None:
            VECTOR_SIZE = len(self.embeddings_df.iloc[:1].values[0])

        def new_list():
            return [[None] for x in range(number_images_select * len(people))]

        saved_images_idx = 0
        # X = np.zeros(((number_images_select * len(people)), VECTOR_SIZE))
        X = new_list()
        y = new_list()
        images_num = new_list()

        for person, images_idx in people.items():
            person_saved_images = 0

            while person_saved_images < number_images_select:
                img_num = images_idx.pop()
                try:
                    img_vector = self._get_embeddings_vector(person, img_num)

                    if img_vector is not None:
                        X[saved_images_idx] = img_vector
                        y[saved_images_idx] = person
                        images_num[saved_images_idx] = img_num

                        saved_images_idx += 1
                        person_saved_images += 1
                finally:
                    pass

        return X, y, people, images_num


# def no_arg(args_name: List[str]):
#     return all(arg not in sys.argv for arg in args_name)


def main():
    global input_folder, models

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", default=input_folder, metavar="path",
                    help="endereco para pasta com as imagens de entrada")
    ap.add_argument("-clf", "--classifier", default="svm", const="svm", nargs='?', choices=models,
                    help=f"classificador responsavel pelo reconhecimento facial ({models})")
    ap.add_argument("-down", "--download", default=False, action='store_true',
                    help="download do banco de imagens lfw")
    ap.add_argument("-rs", "--rand_seed", type=int, default=42,
                    help="seed utilizada na geracao de resultados aleatorios para reproducibilidade")
    ap.add_argument("-ap", "--append", default=False, action='store_true',
                    help="adicionar imagens da pasta de entrada atual a dataframes ja formados")
    ap.add_argument("-si", "--save_images", default=False, action='store_true',
                    help="salvar imagens de faces recortadas")

    training = ap.add_mutually_exclusive_group(required=False)
    training.add_argument("-pt", "--parameter_tuning", default=False, action='store_true',
                          help="otimizacao dos hiperparametros dos classificadores")
    training.add_argument("-oni", "--optimize_num_images", default=False, action='store_true',
                          help="realizacao de testes para detectar numero de imagens ideal")

    ap.add_argument("-ns", "--num_sets", type=int, default=3,
                    # required=(True if not no_arg(["-pt", "--parameter_tuning"]) else False),
                    help="quantidade de sets para divisao dos dados, sendo 1 set para teste e o restante "
                         "para treinamento (usar junto com --parameter_tuning)")
    ap.add_argument("-ipp", "--images_per_person", type=int, default=6,
                    # required=(True if not no_arg(["-pt", "--parameter_tuning"]) else False),
                    help="quantidade de imagens para cada pessoa (valor total que sera dividido entre os sets "
                         "(usar junto com --parameter_tuning))")

    ap.add_argument("-itn", "--images_train", type=int, default=4,
                    # required=(True if no_arg(["-pt", "--parameter_tuning"]) else False),
                    help="quantidade de imagens para treinamento")

    ap.add_argument("-itt", "--images_test", type=int, default=2,
                    # required=(True if no_arg(["-pt", "--parameter_tuning"]) else False),
                    help="quantidade de imagens para teste")

    args = vars(ap.parse_args())

    input_folder = args["input_dir"]

    training = TrainTestRecognition(rand_seed=args["rand_seed"], download_images=args["download"],
                                    append=args["append"], save_images=args["save_images"])

    training.train(classifier=args["classifier"], tune_parameters=args["parameter_tuning"],
                   num_sets=args["num_sets"], images_per_person=args["images_per_person"],
                   optimize_number_images=args["optimize_num_images"],
                   num_images_train=args["images_train"], num_images_test=args["images_test"])


if __name__ == "__main__":
    main()
