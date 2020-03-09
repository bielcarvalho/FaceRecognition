import pickle
import sys
from os import path, makedirs, remove
from time import time
from typing import Optional, Iterable, Tuple, Union, TextIO

import numpy as np
import pandas as pd
from sklearn import neighbors, svm, ensemble, neural_network
from sklearn.metrics import make_scorer, classification_report, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from skopt import BayesSearchCV, callbacks
from skopt.space import Integer, Real, Categorical
import warnings

from tqdm import tqdm

Num_Type = Union[int, float]
Classifier_Type = Union[neighbors.KNeighborsClassifier, svm.SVC, neural_network.MLPClassifier]

# Onde sao armazenados os pkls e os csv com parametros
classifiers_folder = path.join(path.dirname(__file__))
output_folder = path.join(path.dirname(classifiers_folder), "data", "output")

# Onde sao armazenados os csvs com resultados das execucoes de bayes de um respectivo modelo
data_folder: Optional[str] = None

# Arquivo para salvar resultados temporariamente para uso em casos de interrupcao da execucao
temp_logger_path: Optional[str] = None

orig_out_err = sys.stdout, sys.stderr


def to_str(value: object) -> str:
    return str(value).replace(",", "").replace(".", ",")


def list_to_str(list_var: Iterable) -> str:
    return ';'.join(to_str(param_value) for param_value in list_var)


class FixedBayesSearchCV(BayesSearchCV):
    """
    Contorna bug https://github.com/scikit-optimize/scikit-optimize/issues/762
    e evita problemas de compatibilidade entre sklearn 0.21 e skopt.
    Baseado em: https://stackoverflow.com/questions/56609726/bayessearchcv-not-working-because-of-fit-params
    """

    def __init__(self, estimator, search_spaces, optimizer_kwargs=None,
                 n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                 n_points=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score='raise', return_train_score=False):
        """
        See: https://github.com/scikit-optimize/scikit-optimize/issues/762#issuecomment-493689266
        """

        # Bug fix: Added this line
        self.fit_params = fit_params

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)

        # Removed the passing of fit_params to the parent class.
        super(BayesSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, n_jobs=n_jobs, iid=iid,
            refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
            error_score=error_score, return_train_score=return_train_score)

    def _run_search(self, x):
        raise BaseException('Use newer skopt')


class TqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None

    def __init__(self, file: TextIO):
        self.file = file

    def write(self, x: str):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:  # and not any(warning in x for warning in self.suppress):
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


class LoggerCallback(callbacks.DeltaYStopper):
    def __init__(self, model_name: str, num_calls: int, parameter_space: dict, delta: int = 0.0006,
                 n_best: Optional[int] = None):
        global data_folder, temp_logger_path, orig_out_err

        if n_best is None:
            n_best = int(num_calls * 0.25)

        # print(f"Otimizacao sera encerrada quando a diferenca entre os "
        #       f"{n_best} melhores resultados for menor que {delta}")

        super().__init__(delta=delta, n_best=n_best)

        self._bar = tqdm(total=num_calls, desc="Optimizing", unit="iteracoes", file=sys.stdout, dynamic_ncols=True)

        from datetime import datetime
        now = datetime.now()

        temp_logger_path = path.join(data_folder, f"{model_name}_{now.strftime('%Y-%m-%d_%H-%M-%S-%f')}.csv")
        makedirs(path.dirname(temp_logger_path), exist_ok=True)

        orig_out_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = map(TqdmFile, orig_out_err)

        self._best_param = None
        self._best_score = 0

        self.temp_log = open(temp_logger_path, 'w')
        self.temp_log.write(f"{list_to_str(parameter_space.keys())};Score;Time\n")
        self.temp_log.flush()

        import threading
        self.lock = threading.Lock()

        self.last_time = time()

    def close(self):
        try:
            self.temp_log.close()
        except Exception as err:
            pass

    def __del__(self):
        self.close()

    def __call__(self, res):
        curr_param = res.x_iters[-1]
        curr_score = abs(res.func_vals[-1])
        curr_best_score = abs(res.fun)

        self.lock.acquire()
        elapsed_time = time() - self.last_time

        self.temp_log.write(f"{list_to_str(curr_param)};{to_str(curr_score)};{to_str(elapsed_time)}\n")
        self.temp_log.flush()

        if curr_best_score != self._best_score:
            self._best_score = curr_best_score
            self._best_param = res.x
            tqdm.write(f"Best: {self._best_param} ({self._best_score})")
        self._bar.update()

        self.last_time = time()

        self.lock.release()

        # Usar para encerrar por convergencia
        # return super().__call__(res)


class FaceClassifier:
    def __init__(self, rand_seed: Optional[int] = None, parameter_tuning: bool = False,
                 model_folder: str = output_folder, model_name: Optional[str] = None):
        global output_folder

        self.parameter_tuning = parameter_tuning
        self.rand_seed = rand_seed

        self.param_space: Optional[str] = None
        self.model: Optional[Classifier_Type] = None
        self.results_df: Optional[pd.DataFrame] = None

        self.best_idx: Optional[int] = None
        self.param_options: Optional[list] = None

        self.fit_time: Optional[float] = None
        self.score_time: Optional[float] = None

        self.checked_results: Optional[pd.DataFrame] = None

        output_folder = model_folder
        print(f"Arquivos de saida salvos em: {output_folder}")
        makedirs(output_folder, exist_ok=True)

        if model_name is not None:
            if model_name == 'default':
                with open(self._get_model_path("svm"), 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Load models
                with open(self._get_model_path(model_name), 'rb') as f:
                    self.model = pickle.load(f)

    @staticmethod
    def _get_model_path(model_name: str) -> str:
        global classifiers_folder
        return path.join(classifiers_folder, f'{model_name}_classifier.pkl')

    def _tuning(self, model: str, cv: StratifiedKFold, X: Iterable, y: Iterable, bayes: bool = True) -> float:
        """
        Executa parameter tuning
        :param model: nome do classificador
        :param cv: instancia do KFold
        :param X: embeddings
        :param y: labels (nomes das pessoas)
        :param bayes: se otimizacao de bayes deve ser utilizada (bayes == True), ou GridSearchCV (bayes == False)
        :return: melhor score das execucoes
        """
        global orig_out_err

        warnings.filterwarnings("ignore")

        clf, parameter_space, possible_param = self._get_search_space(model)

        score = make_scorer(f1_score, average='weighted')
        print(f"Espaco de busca: {parameter_space}\n")

        if bayes:

            num_iter = 100
            print(f"Maximo de iteracoes na otimizacao de bayes: {num_iter}")

            search = FixedBayesSearchCV(estimator=clf, search_spaces=parameter_space, n_iter=num_iter, cv=cv,
                                        scoring=score, n_jobs=-1, random_state=self.rand_seed)

            callback = LoggerCallback(model, num_iter, parameter_space)

            search.fit(X, y, callback=callback)

            callback.close()
        else:
            # Eh necessario confirmar se o espaco de busca esta em formato aceitavel pelo GridSearchCV
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(estimator=clf, param_grid=parameter_space, cv=cv, scoring=score,
                                  verbose=True, n_jobs=-1)
            search.fit(X, y)

        try:
            warnings.filterwarnings('default')
            sys.stdout, sys.stderr = orig_out_err

        except Exception as err:
            print(err)

        self.model = search.best_estimator_
        self.results_df = pd.DataFrame(search.cv_results_).infer_objects()

        self.param_space = ";".join((str(parameter_space[elem])
                                     if (elem in parameter_space)
                                     else "None")
                                    for elem in possible_param)

        print(search.best_params_, search.best_score_)

        self.best_idx = search.best_index_
        self.param_options = [f"{elem}_options" for elem in possible_param]

        self.fit_time = search.cv_results_["mean_fit_time"][search.best_index_]
        self.score_time = search.cv_results_["mean_score_time"][search.best_index_]

        return search.best_score_

    def _get_search_space(self, model_name: str) -> Tuple[Classifier_Type, dict, list]:
        """
        Retorna campos de busca para cada classificador (a configuracao eh diferente para bayes e gridsearchcv)
        :param model_name: nome do classificador para obter campo de busca
        :return: modelo do classificador, dict com campo de busca, e lista com nomes de parametros usada para compor
        relatorios
        """
        if model_name == "knn":
            parameter_space = {
                'n_neighbors': (1, 6),
                # 'n_neighbors': range(1, 6),
                'algorithm': ['brute', 'ball_tree', 'kd_tree'],
                'weights': ['uniform', 'distance'],
                'leaf_size': (1e+1, 1e+6, 'log-uniform'),
                # 'leaf_size': Categorical([5, 10, 10000, 1000000]),
                'p': (1, 5)
                # 'p': range(1, 6)
            }
            clf = neighbors.KNeighborsClassifier()  # leaf_size=1000000, weights='distance')
            possible_param = ['algorithm', 'leaf_size', 'n_neighbors', 'p', 'weights']

        elif model_name == "mlp":
            parameter_space = {
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['lbfgs', 'adam', 'sgd'],
                'learning_rate': ['constant', 'adaptive'],

                'learning_rate_init': (1e-4, 0.01),
                # 'learning_rate_init': Categorical([0.0005, 0.001, 0.20, 0.23]),
                # 'learning_rate_init': Categorical([1e-10, 1e-8, 1e-6, 0.0001, 0.01]),
                'hidden_layer_sizes': (1000, 2000),
                'alpha': Real(1e-6, 1e-1, 'log-uniform'),
                'tol': Real(1e-10, 1e-1, 'log-uniform')
            }

            clf = neural_network.MLPClassifier(max_iter=400,
                                               # hidden_layer_sizes=2000, alpha=0.000202, solver='lbfgs', tol=2.69e-7,
                                               random_state=self.rand_seed)

            possible_param = ['activation', 'solver', 'learning_rate', 'learning_rate_init',
                              'hidden_layer_sizes', 'alpha', 'tol']

        else:
            parameter_space = {
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'degree': (1, 10),
                # 'degree': list(range(1, 10)),

                'C': (1e-6, 1e+6, 'log-uniform'),
                'gamma': (1e-6, 1e+1, 'log-uniform'),
                # 'gamma': ['auto', 'scale'],
                'tol': (1e-7, 1e-1, 'log-uniform')
            }
            clf = svm.SVC(probability=True,
                          # degree=2,
                          # gamma='auto',
                          # C=100000,
                          random_state=self.rand_seed)
            possible_param = ['kernel', 'degree', 'C', 'gamma', 'tol']

        return clf, parameter_space, possible_param

    def _fit(self, model_name: str, X: Iterable, y: Iterable, num_images_person: int):
        """
        Treina o modelo desejado, e armazena tempo gasto em fit
        :param model_name: nome do classificador para fit
        :param X: embeddings para treinamento
        :param y: labels para treinamento
        :param num_images_person: numero de imagens por pessoa, usado pelo knn para definir numero de vizinhos
        """
        if model_name == 'knn':
            self.model = neighbors.KNeighborsClassifier(algorithm='brute',
                                                        # n_neighbors=max(1, int(num_images_person * 3 / 4)),
                                                        n_neighbors=max(1, int(num_images_person - 1)),
                                                        weights='distance', p=2)

        elif model_name == 'mlp':
            self.model = neural_network.MLPClassifier(activation='logistic', hidden_layer_sizes=1750, max_iter=400,
                                                      alpha=0.000202, solver='lbfgs', tol=2.69e-7,
                                                      random_state=self.rand_seed)

            # self.model = neural_network.MLPClassifier(activation='relu', hidden_layer_sizes=1057, max_iter=400,
            #                                           alpha=0.053, solver='sgd', tol=9e-6,
            #                                           learning_rate="adaptive", learning_rate_init=0.192,
            #                                           random_state=self.rand_seed)

            # self.model = neural_network.MLPClassifier(activation='tanh', hidden_layer_sizes=1367, max_iter=400,
            #                                           learning_rate="constant", alpha=0.0636, solver='sgd', tol=2e-9,
            #                                           learning_rate_init=0.226, random_state=self.rand_seed)

        else:
            self.model = svm.SVC(kernel='poly', degree=2, C=1e-6, gamma=3.33e-5, tol=1e-7, probability=True,
                                 random_state=self.rand_seed)
            # self.model = svm.SVC(kernel='poly', degree=2, C=1e+6, gamma="auto", tol=1e-7, probability=True,
            #                      random_state=self.rand_seed)

        start_time = time()
        self.model.fit(X, y)
        self.fit_time = time() - start_time

    def train(self, X: Iterable, y: Iterable, num_people: int, model_name: str = 'svm',
              num_sets: Optional[Num_Type] = 10, images_per_person: Union[int, Tuple[int, int]] = 10,
              test_images_id: Optional[Iterable] = None, X_test: Optional[Iterable] = None,
              y_test: Optional[Iterable] = None) -> Optional[float]:
        """
        Treinamento e teste
        :param X: embeddings para treinamento ou para serem divididos em sets de treinamento e teste
        :param y: labels (nomes das pessoas) para treinamento ou para serem divididos em sets de treinamento e teste
        :param num_people: numero de pessoas em y
        :param model_name: nome do classificador desejado
        :param num_sets: numero de sets do kfold (para parameter tuning)
        :param images_per_person: numero de imagens para dividir entre os sets, ou
        tupla (numero de imagens treinamento, numero de imagens teste)
        :param test_images_id: id das imagens usadas em teste (opcional)
        :param X_test: embeddings para teste (opcional)
        :param y_test: labels das embeddings para teste (opcional)
        :return: score de teste (melhor score se self.parameter_tuning == True) ou None, se houver apenas treinamento
        """
        global output_folder, temp_logger_path, data_folder

        data_folder = path.join(output_folder, model_name)
        makedirs(data_folder, exist_ok=True)

        score: Optional[float] = None
        scoring_report: Optional[dict] = None
        prob_report: Optional[list] = None
        train_test_images: Optional[tuple] = None

        if self.parameter_tuning is True:
            cv = StratifiedKFold(n_splits=num_sets, shuffle=True, random_state=self.rand_seed)
            score = self._tuning(model_name, cv, X, y)

        else:
            if type(images_per_person) == tuple:
                images_train, images_test = images_per_person
                train_test_images = images_per_person
                images_per_person = images_train + images_test

                if X_test is None and y_test is None and images_test > 0:
                    X, X_test, y, y_test = train_test_split(X, y, stratify=y,
                                                            test_size=images_test*num_people,
                                                            random_state=self.rand_seed)

            else:
                if num_sets > 1:
                    # images_train_person = images_per_person
                    X, X_test, y, y_test = train_test_split(X, y, stratify=y,
                                                            test_size=1 / num_sets,
                                                            random_state=self.rand_seed)

                images_train = (int((num_sets - 1) * images_per_person / num_sets) if num_sets > 1
                                else (num_sets * images_per_person / num_sets))

                train_test_images = (images_train, images_per_person - images_train)

            self._fit(model_name, X, y, num_images_person=images_train)

            if X_test is not None and y_test is not None:
                start_time = time()
                y_pred = self.model.predict(X_test)
                self.score_time = time() - start_time

                y_prob = self.model.predict_proba(X_test)
                chosen = np.argmax(y_prob, axis=1)

                prob_report = [[None] for _ in range(y_pred.size + 1)]
                prob_report[0] = ["Class", "Image_Num", "Prediction", "Probability"]

                for i in range(len(y_pred)):
                    prob_report[i + 1] = [y_test[i], test_images_id[i], y_pred[i], y_prob[i][chosen[i]]]

                scoring_report = classification_report(y_test, y_pred, output_dict=True)

                score = f1_score(y_test, y_pred, average='weighted')

        if output_folder is not None and score is not None:
            self._save_data(images_per_person, model_name, num_people, num_sets, score,
                            scoring_report, prob_report, train_test_images)

        try:
            remove(temp_logger_path)
        except (OSError, TypeError):
            pass

        return score

    def _save_data(self, images_per_person: int, model_name: str, num_people: int, num_sets: Optional[Num_Type],
                   score: float, scoring_report: Optional[dict] = None, prob_report: Optional[list] = None,
                   train_test_images: Optional[Tuple[int, int]] = None):
        """
        Salva dados da execucao com desempenho, e reports de scoring e probabilidade (se houver)
        :param images_per_person: numero de imagens por pessoa usadas na execucao
        :param model_name: nome do classificador usado
        :param num_people: quantidade de pessoas
        :param num_sets: numero de sets
        :param score: metrica de desempenho (do melhor valor no caso de otimizacao de hiperparametros)
        :param scoring_report: dict relacionando o score para cada classe (opcional)
        :param prob_report: lista identificando as classes de teste, as preditas, e o id das imagens (opcional)
        :param train_test_images: tupla (num imagens treinamento, num imagens teste) (opcional)
        """
        global data_folder, output_folder

        try:
            with open(self._get_model_path(model_name), 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as err:
            print(f"Nao foi possivel salvar modelo: {err}")

        file_id = self._save_parameters(images_per_person, model_name, num_people, num_sets, score)

        if scoring_report is not None:
            score_f = path.join(data_folder, f"{model_name}_scoring_{file_id}")
            prob_f = path.join(data_folder, f"{model_name}_probability_{file_id}")

            with open(f"{score_f}.csv", 'w') as f:
                try:
                    key, val = next(iter(scoring_report.items()))
                    f.write(f"Class;{list_to_str(val.keys())}\n")

                    for key, val in scoring_report.items():
                        f.write(f"{key};{list_to_str(val.values())}\n")

                except (TypeError, AttributeError):
                    pass

            if prob_report is not None:
                with open(f"{prob_f}.csv", 'w') as f:
                    for row in prob_report:
                        f.write(f"{list_to_str(row)}\n")

            executions_path = path.join(output_folder, f"executions.csv")
            if not path.exists(executions_path):
                executions_file = open(executions_path, "w")
                executions_file.write("model;execution_id;images_train;images_test;number_people;rand_seed;"
                                      "accuracy;weighted_precision;weighted_recall;weighted_f1-score;support;"
                                      "fit_time;score_time;parameters\n")
            else:
                executions_file = open(executions_path, 'a')

            try:
                train_images, test_images = train_test_images
            except TypeError:
                train_images, test_images = None, None

            executions_file.write(f"{model_name};{file_id};{train_images};{test_images};{num_people};"
                                  f"{self.rand_seed};{to_str(scoring_report['accuracy'])};"
                                  f"{list_to_str(scoring_report['weighted avg'].values())};{self.fit_time};"
                                  f"{self.score_time};{self.model.get_params()}\n")

            executions_file.close()

    def _save_parameters(self, images_per_person: int, model_name: str, num_people: int,
                         num_sets: int, score: float) -> Union[int, str]:
        """
        Salva os parametros da execucao para cada modelo, junto com desempenho e espaco de busca usado (se houver)
        :param images_per_person: numero de imagens por pessoa usadas na execucao
        :param model_name: nome do classificador usado
        :param num_people: quantidade de pessoas
        :param num_sets: numero de sets
        :param score: metrica de desempenho (do melhor valor no caso de otimizacao de hiperparametros)
        :return: id da execucao, que identifica a execucao na planilha de parametros, e os csvs com reports
        """
        global data_folder, output_folder

        parameters_path = path.join(output_folder, f"{model_name}_parameters.csv")
        parameters = self.model.get_params()

        def param_row() -> str:
            row = f"{file_id};{list_to_str(list(parameters.values()))};{images_per_person};" \
                  f"{num_people};{num_sets};{str(self.rand_seed)};{to_str(score)};" \
                  f"{to_str(self.fit_time)};{to_str(self.score_time)}"

            if self.results_df is not None:
                return row + f";{len(self.results_df)};{self.best_idx};{self.param_space}"
            else:
                return row

        try:
            if not path.exists(parameters_path):
                parameters_file = open(parameters_path, "w")
                parameters_file.write("execution_id;" + ';'.join(param_name for param_name in parameters.keys()) +
                                      ";images_per_person;number_people;number_sets;rand_seed;score")

                if self.param_options is not None:
                    parameters_file.write(";mean_fit_time;mean_score_time;iterations;best_idx;"
                                          + ";".join(self.param_options))

                file_id = 1
                parameters_file.write("\n")

            else:
                with open(parameters_path) as f:
                    last_res_id = f.readlines()[-1].split(";", maxsplit=1)[0]
                    file_id = int(last_res_id) + 1

                parameters_file = open(parameters_path, 'a')

            parameters_file.write(param_row())
            # if parameter_tuning:
            if self.results_df is not None:
                self.results_df.to_csv(path.join(data_folder, f"{model_name}_bayes_{file_id}.csv"),
                                       decimal=",", sep=";")
                graph = self.results_df.plot(figsize=(25, 25)).get_figure()
                graph.savefig(path.join(data_folder, f"{model_name}_tuning_{file_id}.jpg"))

                if self.checked_results is not None:
                    self.checked_results.to_csv(path.join(data_folder, f"{model_name}_grid_{file_id}.csv"),
                                                decimal=",", sep=";")
            parameters_file.write("\n")
            parameters_file.close()

        except (OSError, RuntimeError) as err:
            if self.results_df is not None:
                from datetime import datetime
                file_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

                print(f"Nao foi possivel abrir csv de parametros, salvando resultados como {file_id}")
                print(param_row())

            else:
                raise err

        return file_id

    def classify(self, embedding: Iterable) -> Tuple[str, float]:
        """
        Classificar apenas uma imagem
        :param embedding: vetor representando a imagem para classificacao
        :return: nome da classe e probabilidade
        """
        if self.model is None:
            print('Treine um classificador para fazer reconhecimento')
            return

        pred = self.model.predict([embedding])

        # Para knn, a probabilidade eh igual a 1
        prob = np.ravel(self.model.predict_proba([embedding]))

        # prob[::-1].sort()

        return pred[0], round(np.amax(prob), 2)
