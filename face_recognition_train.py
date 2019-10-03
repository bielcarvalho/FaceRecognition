import argparse
import pickle
import sys
from random import shuffle

import cv2
from os import path, makedirs, scandir, removedirs, rmdir
import numpy as np
import pandas as pd
from tqdm import tqdm

from embeddings.FaceEmbeddings import FaceEmbeddings
from detection.FaceDetector import FaceDetector
from classifier.FaceClassifier import FaceClassifier, models

project_data = path.join(path.abspath(path.curdir), "data")

input_folder = path.join(project_data, "input")
output_folder = path.join(project_data, "output")

people_folders = None
number_imgs_list = []
embeddings = []
embeddings_ids = []

embeddings_df = None
people_df = None

face_detector = FaceDetector()
face_recognition = FaceEmbeddings()
face_classifier = FaceClassifier()

vector_size = face_recognition.get_embedding_size()
# vector_size = None


def download(file_path, url):
    import requests

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
    temp_folder = path.join(project_data, "temp")

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
    global embeddings_df, embeddings, embeddings_ids
    index = pd.MultiIndex.from_tuples(embeddings_ids, names=["Name", "Image_Number"])
    embeddings_df = pd.DataFrame(embeddings,
                                 columns=[("v" + str(i)) for i in range(vector_size)],
                                 index=index)
    if path.exists(output_folder):
        # embeddings_df.to_csv(path.join(output_folder, "embeddings.csv"), sep=";")
        with open(path.join(output_folder, "embeddings.pkl"), 'wb') as f:
            pickle.dump(embeddings_df, f)
    del embeddings, embeddings_ids


def people_to_df():
    global number_imgs_list, people_df
    people_df = pd.DataFrame(number_imgs_list, columns=["Name", "Number_Images", "Not_Found"])
    people_df.set_index("Name", inplace=True)
    if path.exists(output_folder):
        with open(path.join(output_folder, "people.pkl"), 'wb') as f:
            pickle.dump(people_df, f)
        # people_df.to_csv(path.join(output_folder, "people.csv"), sep=";")
    del number_imgs_list


def load_dfs():
    global people_df, embeddings_df, people_folders

    people_file = path.join(output_folder, "people.csv")
    embeddings_file = path.join(output_folder, "embeddings.csv")

    people_file = path.join(output_folder, "people.pkl")
    embeddings_file = path.join(output_folder, "embeddings.pkl")
    if not path.exists(people_file) and not path.exists(embeddings_file):
        return False

    # people_df = pd.read_csv(people_file, index_col="Name", sep=";")
    # with open(path.join(output_folder, "people.pkl"), 'wb') as f:
    #     pickle.dump(people_df, f)
    with open(people_file, 'rb') as f:
        people_df = pickle.load(f) #pd.read_csv(people_pkl, index_col="Name", sep=";")
    with open(embeddings_file, 'rb') as f:
        embeddings_df = pickle.load(f) #pd.read_csv(embeddings_file, index_col=["Name", "Image_Number"], sep=";")
    # embeddings_df = pd.read_csv(embeddings_file, index_col=["Name", "Image_Number"], sep=";")
    # with open(path.join(output_folder, "embeddings.pkl"), 'wb') as f:
    #     pickle.dump(embeddings_df, f)

    people_folders = [f.path for f in scandir(input_folder) if f.is_dir()]

    if len(people_folders) > len(people_df):
        return False
    return True


def detect_faces():
    global people_folders, vector_size

    tqdm.write("\nExecutando deteccao facial")

    if people_folders is None:
        people_folders = [f.path for f in scandir(input_folder) if f.is_dir()]
    assert len(people_folders) >= 1

    prog_bar = tqdm(total=len(people_folders), unit="pessoas")

    for person_path in people_folders:

        person_name = path.basename(person_path)
        person_imgs_path = [f.path for f in scandir(person_path) if f.is_file()]

        # if output_folder is not None:
        #     curr_output = path.join(output_folder, person_name)
        #     makedirs(curr_output, exist_ok=True)

        number_imgs_list.append([person_name, len(person_imgs_path) * 2, 0])

        for i in range(len(person_imgs_path)):
            img_path = person_imgs_path[i]
            img = cv2.imread(img_path)

            if img is None:
                tqdm.write('Open image file failed: ' + img_path)
                number_imgs_list[-1][-1] += 2
                continue

            cropped_face = face_detector.extract_face(img)
            if cropped_face is None:
                tqdm.write('No face found in ' + img_path)
                number_imgs_list[-1][-1] += 2
                continue

            cropped_face_flip = cv2.flip(cropped_face, 1)

            embeddings.append(face_recognition.describe(cropped_face))
            embeddings.append(face_recognition.describe(cropped_face_flip))

            embeddings_ids.append([person_name, 2 * i])
            embeddings_ids.append([person_name, (2 * i) + 1])

            # face1 = path.basename(img_path)
            # name, ext = face1.split(".")
            # face2 = name + "_Flip." + ext
            #
            # cv2.imwrite(path.join(person_fold, face1), cropped_face)
            # cv2.imwrite(path.join(person_fold, face2), cropped_face_flip)
        prog_bar.update(1)
    prog_bar.close()

    if output_folder is not None:
        makedirs(output_folder, exist_ok=True)
    vector_size = face_recognition.get_embedding_size()
    embeddings_to_df()
    people_to_df()


def df_tolist(df):
    return [[index] + value for index, value in zip(df.index.tolist(), df.values.tolist())]


def get_feature_vector(person_name, img_number):
    try:
        return embeddings_df.loc[(person_name, img_number)].values
    except KeyError as kerr:
        print(kerr)
        return None
    except TypeError as terr:
        print(terr)
        print(f"ID desejado: {person_name}; Img: {img_number}")
        return None


def get_random_images(images_per_person):
    people_list = df_tolist(people_df.loc[(people_df["Number_Images"] - people_df["Not_Found"]) >= images_per_person])
    assert len(people_list) > 0, "Nao ha pessoas com a quantidade de imagens desejada"

    X = np.zeros(((images_per_person * len(people_list)), vector_size))
    # Y = np.zeros(((images_per_person * len(people_list)), 1))
    Y = [[None] for x in range(images_per_person * len(people_list))]

    tqdm.write(f"\n{len(people_list)} com mais de {images_per_person} imagens")
    progress_bar = tqdm(total=len(people_list * len(people_list)), unit="imagens")

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

                if person_saved_images == images_per_person:
                    break
                progress_bar.update(1)
    progress_bar.close()
    return X, Y


def main():
    global input_folder, output_folder

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", default=input_folder,
                    help="endereco para pasta com as imagens de entrada")
    ap.add_argument("-o", "--output_dir", default=output_folder,
                    help="endereco para a pasta de saida")
    ap.add_argument("-cl", "--classifier", default="ALL",
                    help="classificador responsavel pelo reconhecimento facial (eg: MLP, KNN, SVM, etc)")
    ap.add_argument("-ns", "--num_sets", default=6,
                    help="quantidade de sets para divisao (1 set para teste e o restante para treinamento) "
                         "(para realizar apenas treinamento, colocar 1)")
    ap.add_argument("-ipp", "--images_per_person", default=12,
                    help="quantidade de imagens para cada pessoa")
    ap.add_argument("-pt", "--parameter_tuning", default=True,
                    help="otimizacao dos hiperparametros dos classificadores")
    ap.add_argument("-kf", "--kfold", default=False,
                    help="realizar testes com k-fold (automatico para parameter_tuning)")
    ap.add_argument("-down", "--download", default=False, help="download do banco de imagens lfw")
    args = vars(ap.parse_args())

    input_folder = args["input_dir"]
    output_folder = args["output_dir"]
    if args["download"] is True:
        down_img_db()
    assert path.exists(input_folder), f"A pasta {input_folder} nao existe, informe uma pasta valida"

    if args["parameter_tuning"] or args["kfold"]:
        assert args["num_sets"] > 1, f"Para cross-validation, e que haja sets de treinamento e testes " \
                                     f"(num_sets >= 2)"
        assert (args["images_per_person"] >= args["num_sets"]) and \
               (args["images_per_person"] % args["num_sets"] == 0), \
            f"Deve haver ao menos uma imagem por set para o cross-validation"

    classifiers = []
    clf_name = args["classifier"].lower()
    if clf_name == "all":
        classifiers = models
    else:
        assert clf_name in models, f"O classificador {clf_name} nao esta " \
                                   f"implementado, tente algum desses: {models}"
        classifiers.append(clf_name)

    if load_dfs() is False:
        detect_faces()
    X, Y = get_random_images(args["images_per_person"])

    for model in classifiers:
        print("Training Model {}".format(model))
        face_classifier.train(X, Y, model=model, num_sets=args["num_sets"], k_fold=args["kfold"],
                              hyperparameter_tuning=args["parameter_tuning"], images_per_person=args["images_per_person"],
                              save_model_path=f'./classifier/{model}_classifier.pkl')
                              # save_model_path=f'{output_folder}/classifier/{model}_classifier.pkl')


if __name__ == "__main__":
    main()
