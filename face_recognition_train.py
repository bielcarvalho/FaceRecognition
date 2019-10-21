import argparse
import sys
from os import path, makedirs, scandir
from random import shuffle

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from classifier.FaceClassifier import FaceClassifier

models = [
    "random_forest",
    "knn",
    "svm",
    "mlp",
    "dtree",
    "all"
]

input_folder = path.join(path.abspath(path.curdir), "data", "input")

people_folders = None
number_imgs_list = []
embeddings = []
embeddings_ids = []

embeddings_df = None
people_df = None

vector_size = None

face_classifier = FaceClassifier()


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
    global people_folders, vector_size, embeddings, embeddings_ids, number_imgs_list

    from embeddings.FaceEmbeddings import FaceEmbeddings
    from detection.FaceDetector import FaceDetector

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
            try:
                img = Image.open(img_path)
            except OSError or IOError:
                tqdm.write('Open image file failed: ' + img_path)
                number_imgs_list[-1][-1] += 1
                continue

            if img is None:
                tqdm.write('Open image file failed: ' + img_path)
                number_imgs_list[-1][-1] += 1
                continue

            image_torch, score = face_detector.extract_face(img)
            # image_torch, score = face_detector.extract_face(img, save_path=path.join(curr_output, str(i) + "a.jpg"))

            if image_torch is None or score < 0.5:
                tqdm.write(f'No face found in {img_path}')
                if score is not None:
                    tqdm.write(f'(Score: {score}')
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


def df_tolist(df):
    return [[index] + value for index, value in zip(df.index.tolist(), df.values.tolist())]


def get_feature_vector(person_name, img_number):
    global embeddings_df
    try:
        return embeddings_df.loc[(person_name, img_number)].values
    except KeyError as kerr:
        tqdm.write(kerr)
        return None
    except TypeError as terr:
        print(terr)
        tqdm.write(f"ID desejado: {person_name}; Img: {img_number}")
        return None


def get_random_images(images_per_person):
    global vector_size, people_df, embeddings_df
    people_list = df_tolist(people_df.loc[(people_df["Number_Images"] - people_df["Not_Found"]) >= images_per_person])
    assert len(people_list) > 0, "Nao ha pessoas com a quantidade de imagens desejada"

    if vector_size is None:
        vector_size = len(embeddings_df.iloc[:1].values[0])

    X = np.zeros(((images_per_person * len(people_list)), vector_size))
    Y = [[None] for x in range(images_per_person * len(people_list))]

    tqdm.write(f"\n{len(people_list)} pessoas com mais de {images_per_person} imagens")
    progress_bar = tqdm(total=(images_per_person * len(people_list)), desc="Sampling", unit="imagens")

    saved_images_idx = 0
    for row in people_list:
        person, num_images, _ = row
        rand_images = list(range(num_images))
        shuffle(rand_images)

        person_saved_images = 0

        for img_num in rand_images:
            img_vector = get_feature_vector(person, img_num)

            if img_vector is not None:
                X[saved_images_idx] = img_vector
                Y[saved_images_idx] = person

                saved_images_idx += 1
                person_saved_images += 1

                progress_bar.update(1)

                if person_saved_images == images_per_person:
                    break

    progress_bar.close()

    del people_df, embeddings_df

    return X, Y, len(people_list)


def main():
    global input_folder, models

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", default=input_folder,
                    help="endereco para pasta com as imagens de entrada")
    ap.add_argument("-clf", "--classifier", default="svm", const="svm", nargs='?', choices=models,
                    help=f"classificador responsavel pelo reconhecimento facial ({models})")
    ap.add_argument("-ns", "--num_sets", type=int, default=5,
                    help="quantidade de sets para divisao dos dados, sendo 1 set para teste e o restante "
                         "para treinamento (para realizar apenas treinamento, colocar 1)")
    ap.add_argument("-ipp", "--images_per_person", type=int, default=20,
                    help="quantidade de imagens para cada pessoa (valor total que sera dividido entre os sets")
    ap.add_argument("-pt", "--parameter_tuning", default=False, action='store_true',
                    help="otimizacao dos hiperparametros dos classificadores")
    ap.add_argument("-kf", "--kfold", type=bool, default=False,
                    help="realizar testes com k-fold (automatico para parameter_tuning)")
    ap.add_argument("-down", "--download", default=False, action='store_true',
                    help="download do banco de imagens lfw")
    args = vars(ap.parse_args())

    input_folder = args["input_dir"]

    if args["download"] is True:
        down_img_db()

    if args["parameter_tuning"] or args["kfold"]:
        assert args["num_sets"] > 1, f"Para cross-validation, e que haja sets de treinamento e testes " \
                                     f"(num_sets >= 2)"
        assert (args["images_per_person"] >= args["num_sets"]) and \
               (args["images_per_person"] % args["num_sets"] == 0), \
            f"Deve haver ao menos uma imagem por set para o cross-validation, e o valor deve ser proporcional ao " \
            f"numero de sets para que as diferentes classes tenham a mesma probabilidade de serem classificadas"

    classifiers = []
    clf_name = args["classifier"].lower()
    if clf_name == "all":
        classifiers = models.copy()
        classifiers.remove("all")
    else:
        classifiers.append(clf_name)

    if load_dfs() is False:
        assert path.exists(input_folder), f"A pasta {input_folder} nao existe, informe uma pasta valida"
        detect_faces()
    X, Y, num_people = get_random_images(args["images_per_person"])

    for model in classifiers:
        print("Training Model {}".format(model))
        face_classifier.train(X, Y, model=model, num_sets=args["num_sets"], k_fold=args["kfold"],
                              parameter_tuning=args["parameter_tuning"],
                              save_model_path=f'./classifier/{model}_classifier.pkl',
                              images_per_person=args["images_per_person"],
                              num_people=num_people)


if __name__ == "__main__":
    main()
