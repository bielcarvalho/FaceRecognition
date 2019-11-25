import argparse
import sys
from os import path, makedirs, scandir
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from classifier.FaceClassifier import FaceClassifier

models = [
    "knn",
    "svm",
    "mlp",
    "all"
]

save_images = False
proj_folder = path.dirname(__file__)
input_folder = path.join(proj_folder, "data", "input")
output_folder = path.join(proj_folder, "data", "output")

people_folders = None
number_imgs_list = []
embeddings = []
embeddings_ids = []

embeddings_df = None
people_df = None

vector_size = None

random_seed: Optional[int] = None


def download(file_path, url):
    import requests
    import math

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length'))
    block_size = 1024

    with open(file_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), desc="Download",
                         unit='B', unit_scale=True, unit_divisor=1024):
            f.write(data)


def down_img_db():
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


def embeddings_to_df():
    global embeddings_df, embeddings, embeddings_ids, vector_size

    index = pd.MultiIndex.from_tuples(embeddings_ids, names=["Name", "Image_Number"])

    embeddings_df = pd.DataFrame(embeddings,
                                 columns=[("v" + str(i)) for i in range(vector_size)],
                                 index=index)

    if path.exists(input_folder):
        embeddings_df.to_pickle(path.join(input_folder, "embeddings.bz2"))
        # embeddings_df.to_csv(path.join(input_folder, "embeddings.csv"), sep=";")
    del embeddings, embeddings_ids


def people_to_df():
    global number_imgs_list, people_df
    people_df = pd.DataFrame(number_imgs_list, columns=["Name", "Number_Images", "Not_Found"])
    people_df.set_index("Name", inplace=True)
    if path.exists(input_folder):
        people_df.to_pickle(path.join(input_folder, "people.bz2"))
        # people_df.to_csv(path.join(input_folder, "people.csv"), sep=";")
    del number_imgs_list


def load_dfs():
    global people_df, embeddings_df, people_folders

    people_file = path.join(input_folder, "people.bz2")
    embeddings_file = path.join(input_folder, "embeddings.bz2")
    if not path.exists(people_file) and not path.exists(embeddings_file):
        return False

    people_df = pd.read_pickle(people_file)
    embeddings_df = pd.read_pickle(embeddings_file)

    people_folders = [f.path for f in scandir(input_folder) if f.is_dir()]

    if len(people_folders) > len(people_df):
        return False
    return True


def detect_faces():
    global people_folders, vector_size, embeddings, embeddings_ids, number_imgs_list, save_images

    from embeddings.FaceEmbeddings import FaceEmbeddings
    from detection.FaceDetector import FaceDetector
    import cv2

    face_detector = FaceDetector()
    face_recognition = FaceEmbeddings()
    vector_size = face_recognition.get_embedding_size()

    print("\nExecutando deteccao facial")

    if people_folders is None:
        people_folders = [f.path for f in scandir(input_folder) if f.is_dir()]
    assert len(people_folders) >= 1

    prog_bar = tqdm(total=len(people_folders), desc="Detectando", position=0, unit="pessoas")

    for person_path in people_folders:

        person_name = path.basename(person_path)
        person_imgs_path = [f.path for f in scandir(person_path) if f.is_file()]

        # if output_folder is not None:
        #     curr_output = path.join(output_folder, person_name)
        #     makedirs(curr_output, exist_ok=True)

        number_imgs_list.append([person_name, len(person_imgs_path), 0])

        for i, img_path in enumerate(person_imgs_path):

            # face_img = path.join(person_path, "MTCNN", f"{str(i)}.jpg")
            # if path.exists(face_img):
            #     import cv2
            #     img = cv2.imread(face_img)
            #
            #     img = face_detector.pre_process(img)
            #
            #     embeddings.append(face_recognition.describe(img))
            #     embeddings_ids.append([str(person_name), i])
            #
            #     continue

            try:
                # img = Image.open(img_path)
                img = cv2.imread(img_path, )

            except (OSError, IOError):
                tqdm.write('Open image file failed: ' + img_path)
                number_imgs_list[-1][-1] += 1
                continue

            if img is None:
                tqdm.write('Open image file failed: ' + img_path)
                number_imgs_list[-1][-1] += 1
                continue

            tqdm.write(f'Detecting image {i}, file: {img_path}')

            # image_torch, score = face_detector.extract_face(img)
            image_torch, score = face_detector.extract_face(img, save_path=(path.join(person_path, "MTCNN",
                                                                                      f"{str(i)}.jpg")
                                                                            if save_images is True else None))

            if image_torch is None or score < 0.5:
                tqdm.write(f'No face found in {img_path}')
                if score is not None:
                    tqdm.write(f'(Score: {score})')
                number_imgs_list[-1][-1] += 1
                continue

            embeddings.append(face_recognition.describe(image_torch))
            embeddings_ids.append([str(person_name), i])
        prog_bar.update(1)
    prog_bar.close()

    vector_size = face_recognition.get_embedding_size()
    embeddings_to_df()
    people_to_df()

    del people_folders


def df_tolist(df: pd.DataFrame) -> list:
    return [[index] + value for index, value in zip(df.index.tolist(), df.values.tolist())]


def get_feature_vector(person_name: str, img_number: int) -> Optional[np.ndarray]:
    global embeddings_df
    try:
        return embeddings_df.loc[(person_name, img_number)].values
    except (KeyError, TypeError) as kerr:
        # tqdm.write(kerr)
        # tqdm.write(f"ID desejado: {person_name}; Img: {img_number}")
        return None


def get_shuffled_idxs(images_per_person: int) -> dict:
    global people_df, random_seed
    people_list = df_tolist(people_df.loc[(people_df["Number_Images"] - people_df["Not_Found"]) >= images_per_person])
    assert len(people_list) > 0, "Nao ha pessoas com a quantidade de imagens desejada"

    shuffled_idxs = {}

    import random

    # if random_seed is not None:
    #     print(f"Teste com {people_list[0]}")
    #     person_test, num_images_test, _ = people_list[0]
    #     for i in range(5):
    #         random.seed(random_seed)
    #         temp_list = list(range(num_images_test))
    #         random.shuffle(temp_list)
    #         print(temp_list)

    for person, num_images, not_found in people_list:
        # person, num_images, _ = row
        shuffled_idxs[person] = list(range(num_images))

        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(shuffled_idxs[person])

    print(f"Indices de {people_list[0][0]}: {shuffled_idxs[people_list[0][0]]}")

    return shuffled_idxs


def min_images_test(num_images_test=5, num_images_train=10):
    global models, random_seed

    try:
        shuffled_idxs = get_shuffled_idxs(images_per_person=(num_images_test + num_images_train))
    except AssertionError:
        raise Exception("Nao ha pessoas com a quantidade de imagens necessaria para efetuar o teste")

    classifiers = models.copy()
    try:
        classifiers.remove("all")
    except KeyError:
        pass
    # print(type(images_idx))
    X_test, y_test, shuffled_idxs, images_num = select_embeddings_aux(num_images_test, shuffled_idxs)
    X_train, y_train, temp = [], [], []

    face_classifier = FaceClassifier(random_seed)
    # print(type(images_idx))

    progress_bar = tqdm(total=num_images_train, desc="Running tests",
                        unit="iteration", file=sys.stdout,
                        dynamic_ncols=True)

    for i in range(1, num_images_train + 1):
        X_train, y_train, shuffled_idxs, temp = select_embeddings_aux(1, shuffled_idxs, X_train, y_train, temp)

        best_score, best_model = 0.0, None

        for model in classifiers:
            new_score = face_classifier.train(X=X_train, y=y_train, X_test=X_test, y_test=y_test, model_name=model,
                                              num_sets=i / (i + num_images_test),
                                              images_per_person=(i, num_images_test),
                                              num_people=len(shuffled_idxs),
                                              test_images_id=images_num)

            if new_score > best_score:
                best_score, best_model = new_score, model

        progress_bar.write(f"Melhor com {i} imagens - {best_model}: {best_score}")
        progress_bar.update()


def get_random_images(images_per_person: int):
    global vector_size, people_df, embeddings_df
    # people_list = df_tolist(people_df.loc[(people_df["Number_Images"] - people_df["Not_Found"]) >= images_per_person])
    # assert len(people_list) > 0, "Nao ha pessoas com a quantidade de imagens desejada"
    #
    # if vector_size is None:
    #     vector_size = len(embeddings_df.iloc[:1].values[0])
    #
    # X = np.zeros(((images_per_person * len(people_list)), vector_size))
    # Y = [[None] for x in range(images_per_person * len(people_list))]
    #
    # tqdm.write(f"\n{len(people_list)} pessoas com mais de {images_per_person} imagens")
    # progress_bar = tqdm(total=(images_per_person * len(people_list)), desc="Sampling", unit="imagens")
    #
    # saved_images_idx = 0
    # for row in people_list:
    #     person, num_images, _ = row
    #     rand_images = list(range(num_images))
    #     shuffle(rand_images)
    #
    #     person_saved_images = 0
    #
    #     for img_num in rand_images:
    #         img_vector = get_feature_vector(person, img_num)
    #
    #         if img_vector is not None:
    #             X[saved_images_idx] = img_vector
    #             Y[saved_images_idx] = person
    #
    #             saved_images_idx += 1
    #             person_saved_images += 1
    #
    #             progress_bar.update(1)
    #
    #             if person_saved_images == images_per_person:
    #                 break
    #
    # progress_bar.close()
    #
    # del people_df, embeddings_df
    #
    # return X, Y, len(people_list)

    shuffled_idx = get_shuffled_idxs(images_per_person)

    print(f"\nSelecionando {len(shuffled_idx) * images_per_person} imagens "
          f"de {len(shuffled_idx)} pessoas com mais de {images_per_person} imagens")

    X, Y, shuffled_idx, images_num = select_embeddings_aux(images_per_person, shuffled_idx)

    del people_df, embeddings_df

    return X, Y, len(shuffled_idx), images_num


def select_embeddings_aux(images_per_person: int, people: dict, X: Optional[list] = None, y: Optional[list] = None,
                          images_num: Optional[list] = None) -> Tuple[Union[list, np.ndarray], list, dict, list]:
    global vector_size, embeddings_df
    if vector_size is None:
        vector_size = len(embeddings_df.iloc[:1].values[0])

    def new_list():
        return [[None] for x in range(images_per_person * len(people))]

    if X is not None and y is not None:
        saved_images_idx = len(X)
        X.extend(new_list())
        y.extend(new_list())
        images_num.extend(new_list())
    else:
        saved_images_idx = 0
        X = np.zeros(((images_per_person * len(people)), vector_size))
        y = new_list()
        images_num = new_list()

    # tqdm.write(f"\n{len(people)} pessoas com mais de {images_per_person} imagens")
    # progress_bar = tqdm(total=(images_per_person * len(people)), desc="Sampling", unit="imagens")

    # print(next(iter(people.items())))

    for person, images_idx in people.items():
        person_saved_images = 0

        while person_saved_images < images_per_person:
            img_num = images_idx.pop()
            try:
                img_vector = get_feature_vector(person, img_num)

                if img_vector is not None:
                    X[saved_images_idx] = img_vector
                    y[saved_images_idx] = person
                    images_num[saved_images_idx] = img_num

                    saved_images_idx += 1
                    person_saved_images += 1
            finally:
                pass

    #             progress_bar.update(1)
    # progress_bar.close()
    # print(next(iter(people.items())))

    return X, y, people, images_num


def main():
    global input_folder, models, random_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", default=input_folder,
                    help="endereco para pasta com as imagens de entrada")
    ap.add_argument("-clf", "--classifier", default="svm", const="svm", nargs='?', choices=models,
                    help=f"classificador responsavel pelo reconhecimento facial ({models})")
    ap.add_argument("-ns", "--num_sets", type=int, default=5,
                    help="quantidade de sets para divisao dos dados, sendo 1 set para teste e o restante "
                         "para treinamento (para realizar apenas treinamento, colocar 1)")
    ap.add_argument("-ipp", "--images_per_person", type=int, default=10,
                    help="quantidade de imagens para cada pessoa (valor total que sera dividido entre os sets")
    ap.add_argument("-pt", "--parameter_tuning", default=False, action='store_true',
                    help="otimizacao dos hiperparametros dos classificadores")
    ap.add_argument("-kf", "--kfold", type=bool, default=False,
                    help="realizar testes com k-fold (automatico para parameter_tuning)")
    ap.add_argument("-down", "--download", default=False, action='store_true',
                    help="download do banco de imagens lfw")
    ap.add_argument("-rs", "--rand_seed", type=int, default=42,
                    help="seed utilizada na geracao de resultados aleatorios para reproducibilidade")
    ap.add_argument("-tmi", "--test_min_images", default=False, action='store_true',
                    help="realizacao de testes para detectar numero de imagens ideal")
    args = vars(ap.parse_args())

    if args["rand_seed"] >= 0:
        random_seed = args["rand_seed"]
        print(f"Reproducibilidade possivel com seed {random_seed}")

    input_folder = args["input_dir"]

    if args["download"] is True:
        down_img_db()

    classifiers = []
    if not args["test_min_images"]:
        if args["parameter_tuning"] or args["kfold"]:
            assert args["num_sets"] > 1, f"Para cross-validation, e que haja sets de treinamento e testes " \
                                         f"(num_sets >= 2)"
            assert (args["images_per_person"] >= args["num_sets"]) and \
                   (args["images_per_person"] % args["num_sets"] == 0), \
                f"Deve haver ao menos uma imagem por set para o cross-validation, e o valor deve ser proporcional ao " \
                f"numero de sets para que as diferentes classes tenham a mesma probabilidade de serem classificadas"

        clf_name = args["classifier"].lower()
        if clf_name == "all":
            classifiers = models.copy()
            classifiers.remove("all")
        else:
            classifiers.append(clf_name)

    if load_dfs() is False:
        assert path.exists(input_folder), f"A pasta {input_folder} nao existe, informe uma pasta valida"
        detect_faces()

    if args["test_min_images"]:
        min_images_test()

    else:
        X, Y, num_people, images_test_ids = get_random_images(args["images_per_person"])

        face_classifier = FaceClassifier(random_seed, args["parameter_tuning"])

        for model in classifiers:
            print("Training Model {}".format(model))
            face_classifier.train(X, Y, model_name=model, num_sets=args["num_sets"],
                                  images_per_person=args["images_per_person"],
                                  num_people=num_people,
                                  test_images_id=images_test_ids)


if __name__ == "__main__":
    main()
