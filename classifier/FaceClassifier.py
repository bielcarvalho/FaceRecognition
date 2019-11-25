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
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV, callbacks
from skopt.space import Integer, Real, Categorical
import warnings

from tqdm import tqdm

PATH_TO_PKL = 'trained_classifier.pkl'
Num_Type = Union[int, float]
Classifier_Type = Union[neighbors.KNeighborsClassifier, svm.SVC, neural_network.MLPClassifier]

# Onde sao armazenados os pkls e os csv com parametros
output_folder = path.join(path.dirname(__file__))
# output_folder = path.join(path.dirname(__file__), "data", "output")
# output_folder = path.abspath(f'./classifier/')
# output_folder = path.join(path.curdir, "data", "output")

# Onde sao armazenados os csvs com resultados das execucoes de bayes de um respectivo modelo
data_folder: Optional[str] = None

# Arquivo usado para lidar com necessidade de interromper treinamento por configuracao demorada
temp_logger_path: Optional[str] = None

orig_out_err = sys.stdout, sys.stderr


def to_str(value: object) -> str:
    return str(value).replace(",", "").replace(".", ",")


def list_to_str(list_var: Iterable) -> str:
    return ';'.join(to_str(param_value) for param_value in list_var)


class FixedBayesSearchCV(BayesSearchCV):
    """
    Baseado em: https://stackoverflow.com/questions/56609726/bayessearchcv-not-working-because-of-fit-params

    Dirty hack to avoid compatibility issues with sklearn 0.2 and skopt.
    Credit: https://www.kaggle.com/c/home-credit-default-risk/discussion/64004

    For context, on why the workaround see:
        - https://github.com/scikit-optimize/scikit-optimize/issues/718
        - https://github.com/scikit-optimize/scikit-optimize/issues/762
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

        print(f"Otimizacao sera encerrada quando a diferenca entre os "
              f"{n_best} melhores resultados for menor que {delta}")

        super().__init__(delta=delta, n_best=n_best)

        self._bar = tqdm(total=num_calls, desc="Optimizing", unit="iteracoes", file=sys.stdout, dynamic_ncols=True)

        from datetime import datetime
        now = datetime.now()

        temp_logger_path = path.join(data_folder, f"{model_name}_{now.strftime('%Y-%m-%d_%H-%M-%S-%f')}.csv")
        makedirs(path.dirname(temp_logger_path), exist_ok=True)

        orig_out_err = sys.stdout, sys.stderr

        # tqdm_output = TqdmFile(file=self._bar)
        # sys.stdout = TqdmFile(sys.stdout)
        # sys.stderr = tqdm_output
        sys.stdout, sys.stderr = map(TqdmFile, orig_out_err)

        self._best_param = None
        self._best_score = 0

        # with open(temp_logger_path, 'w') as f:
        #     f.write(f"{list_to_str(parameter_space.keys())};Score;Time\n")

        self.temp_log = open(temp_logger_path, 'w')
        self.temp_log.write(f"{list_to_str(parameter_space.keys())};Score;Time\n")
        self.temp_log.flush()

        import threading
        self.lock = threading.Lock()

        self.last_time = time()
        # self.total_time = 20 * 60 * num_calls
        # self.time_spent = 0

        self.slow_counter = 0  # > 1 hour

    def close(self):
        try:
            self.temp_log.close()
        finally:
            pass

    def __del__(self):
        self.close()

    def __call__(self, res):
        # pprint(res)
        # self._best_model = res.func_vals[-1]

        curr_param = res.x_iters[-1]
        curr_score = abs(res.func_vals[-1])
        curr_best_score = abs(res.fun)

        self.lock.acquire()
        elapsed_time = time() - self.last_time

        self.temp_log.write(f"{list_to_str(curr_param)};{to_str(curr_score)};{to_str(elapsed_time)}\n")
        self.temp_log.flush()
        if elapsed_time >= 3600:
            print(f"{curr_param} levou um tempo de {elapsed_time}")
            self.slow_counter += 1

        if curr_best_score != self._best_score:
            self._best_score = curr_best_score
            self._best_param = res.x
            tqdm.write(f"Best: {self._best_param} ({self._best_score})")
        self._bar.update()

        # dump(res, self.checkpoint_path)
        self.last_time = time()

        completed = len(res.func_vals) / self._bar.total

        global output_folder

        # Numero de vezes seguidas em que a configuracao de parametros atual foi utilizada
        # evaluated = res.x_iters.count(curr_param)

        # if (completed > 0.5 and (  # evaluated >= self.n_best or
        #         (path.exists(path.join(output_folder, "awaken.txt")) and
        #          (self.slow_counter >= 2 and (self._bar.total - len(res.func_vals)) >= 5)))):
        #     self.lock.release()
        #     return True

        self.lock.release()

        if completed > 0.75:
            return super().__call__(res)


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
                raise NotImplementedError
                # model_folder = path.join(BASE_DIR, PATH_TO_PKL)
            else:
                # Load models
                with open(self.get_model_path(model_name), 'rb') as f:
                    self.model = pickle.load(f)

    @staticmethod
    def get_model_path(model_name: str) -> str:
        global output_folder
        return path.join(output_folder, f'{model_name}_classifier.pkl')

    def check_optimization_results(self, clf: Classifier_Type, cv: StratifiedKFold,
                                   X: Iterable, y: Iterable, score: float):
        import math
        sorted_res = self.results_df.sort_values(by='mean_test_score', ascending=False, inplace=False)

        best_values = sorted_res.iloc[:min(5, sorted_res.shape[0]), (cv.get_n_splits() + 7):-1]
        grid_param = {}
        unique_values = 0
        iterations = 1

        print(best_values)
        from pandas.api.types import is_float_dtype, is_string_dtype, is_object_dtype

        def col_range(col_name):
            return best_values[col_name].max() - best_values[col_name].min()

        def round_order_magn(float_var):
            # temp_exp = round(math.log10(float_var))
            # print(temp_exp, int(temp_exp))
            return round(math.log10(float_var))

        for col in best_values.columns:
            if is_string_dtype(best_values[col]) or is_object_dtype(best_values[col]):
                param = list(sorted_res[col].unique())
                print(param)
            else:
                param = list(best_values[col].unique())
                print(param)

                # if is_float_dtype(best_values[col]) and col_range(col) >= 0.001:
                if is_float_dtype(best_values[col]):
                    # Evitar repeticao de parametros muito proximos (eg: 2.67E-06, 2.69E-06, 2.73E-06)
                    temp_param = set()
                    [temp_param.add(round_order_magn(var)) for var in param]
                    param = [(10 ** var) for var in temp_param]
                print(param)

            # Remover prefixo "param_"
            # param_name = col.strip("param_")
            grid_param[col[6:]] = param

            if len(param) > 1:
                unique_values += 1
                iterations *= len(param)

        print(iterations)

        if unique_values >= 2 and iterations <= 250:
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(estimator=clf, param_grid=grid_param, cv=cv, scoring=score,
                                  verbose=True, n_jobs=-1)
            search.fit(X, y)
            self.checked_results = pd.DataFrame(search.cv_results_).infer_objects()
            print(search.best_params_, search.best_score_)
        print(best_values)
        print(grid_param)

    def tuning(self, model: str, cv: StratifiedKFold, X: Iterable, y: Iterable, bayes: bool = True) -> float:
        global orig_out_err
        print("Parameter tuning")

        warnings.filterwarnings("ignore")

        clf, parameter_space, possible_param = self.get_search_space(model)

        import copy
        clf_copy = copy.deepcopy(clf)

        score = make_scorer(f1_score, average='weighted')
        print(f"Espaco de busca: {parameter_space}\n")

        if bayes:

            num_iter = 100
            print(f"Maximo de iteracoes na otimizacao de bayes: {num_iter}")

            # search = BayesSearchCV(estimator=clf, search_spaces=parameter_space, n_iter=num_iter, cv=cv, verbose=False,
            #                        scoring=score, n_jobs=-1, random_state=self.rand_seed)

            search = FixedBayesSearchCV(estimator=clf, search_spaces=parameter_space, n_iter=num_iter, cv=cv,
                                        scoring=score, n_jobs=-1, random_state=self.rand_seed)

            callback = LoggerCallback(model, num_iter, parameter_space)

            search.fit(X, y, callback=callback)

            callback.close()
        else:
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


        # self.check_optimization_results(clf_copy, cv, X, y, score)

        return search.best_score_

    def get_search_space(self, model_name: str) -> Tuple[Classifier_Type, dict, list]:
        if model_name == "svm":
            parameter_space = {
                'kernel': Categorical(['rbf', 'sigmoid']),
                # 'kernel': ['linear'],
                # , 'rbf', 'sigmoid'],
                # 'kernel': Categorical(['linear', 'rbf', 'sigmoid', 'poly']),
                # 'kernel': Categorical(['poly']),
                # 'degree': (1, 10),
                # 'degree': Categorical([2, 5, 9]),
                # 'degree': Categorical([2]),

                'C': (1e-6, 1e+6, 'log-uniform'),
                # 'C': Categorical([1e-6, 1e+1, 1e+4, 1e+5, 1e+6]),
                # 'C': Categorical([1e-6, 1e+5, 1e+6]),
                # 'gamma': (1e-6, 1e+1, 'log-uniform'),
                # 'gamma': Categorical([1e-7, 1e-6, 1e-5, 'auto', 'scale']),
                'gamma': Categorical(['auto', 'scale']),
                # 'tol': Categorical([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1])
                'tol': Real(1e-7, 1e-1, 'log-uniform')
                # 'tol': Categorical([1e-7, 1e-6, 1e-4, 1e-2, 1e-1])
            }
            clf = svm.SVC(probability=True,
                          # degree=2,
                          # gamma='auto',
                          # C=100000,
                          random_state=self.rand_seed)
            possible_param = ['kernel', 'degree', 'C', 'gamma', 'tol']

        elif model_name == "knn":
            parameter_space = {
                'n_neighbors': (1, 6),
                # 'n_neighbors': range(1, 6),
                # 'algorithm': ['brute'],
                'algorithm': ['brute', 'ball_tree', 'kd_tree'],
                'weights': ['uniform', 'distance'],
                # 'leaf_size': (1e+1, 1e+6, 'log-uniform'),
                # 'leaf_size': Categorical([5, 10, 10000, 1000000]),
                # 'leaf_size': [1000000],
                'p': (1, 5)
                # 'p': range(1, 6)
            }
            clf = neighbors.KNeighborsClassifier()  # leaf_size=1000000, weights='distance')
            possible_param = ['algorithm', 'leaf_size', 'n_neighbors', 'p', 'weights']

        elif model_name == "mlp":
            parameter_space = {
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['lbfgs'],
                # 'solver': ['adam'],
                # 'solver': Categorical(['sgd']),
                # 'learning_rate': Categorical(['constant', 'adaptive']),

                # 'learning_rate_init': Real(1e-4, 0.01),
                # 'learning_rate_init': Categorical([0.0005, 0.001, 0.20, 0.23]),

                # 'learning_rate_init': Categorical([1e-10, 1e-8, 1e-6, 0.0001, 0.01]),
                # 'hidden_layer_sizes': Integer(1500, 2000),
                # 'alpha': Categorical([1e-10, 1e-5, 0.0001, 0.005, 0.05]),
                # 'alpha': Categorical([1e-10]),
                # 'alpha': [0.0002],
                'alpha': Real(1e-4, 1e-2, 'log-uniform'),
                'tol': Real(1e-10, 1e-6, 'log-uniform')
                # 'tol': [1e-7]
            }

            clf = neural_network.MLPClassifier(hidden_layer_sizes=2000, max_iter=400,
                                               # alpha=0.000202, solver='lbfgs', tol=2.69e-7,
                                               random_state=self.rand_seed)

            # clf = neural_network.MLPClassifier(max_iter=400,
            #                                    # learning_rate='constant',
            #                                    # learning_rate_init=0.05,
            #                                    random_state=self.rand_seed)
            possible_param = ['activation', 'solver', 'learning_rate', 'learning_rate_init',
                              'hidden_layer_sizes', 'alpha', 'tol']

        elif model_name == "dtree":
            parameter_space = {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_split': (2, 50),
                'min_samples_leaf': (1, 20),
                # 'max_depth': (50, 500),
                # 'max_leaf_nodes': (50, 500),
                # 'min_impurity_decrease': (1e-10, 1e-6)
                'max_features': ["auto", "sqrt", "log2", None],
                # 'max_features': (0.1, 0.9)
            }
            clf = DecisionTreeClassifier()
            possible_param = ['criterion', 'splitter', 'min_samples_split', 'min_samples_leaf',
                              'max_depth', 'max_leaf_nodes', 'min_impurity_decrease', 'max_features']
        else:
            parameter_space = {
                'n_estimators': (50, 200),
                'criterion': ['gini', 'entropy'],
                # 'min_samples_split': (2, 20),
                # 'min_samples_leaf': (1, 20),
                # 'max_depth': (2, 150)
                # 'max_features': (0.1, 0.9)
            }
            clf = ensemble.RandomForestClassifier()
            possible_param = ['criterion', 'n_estimators', 'min_samples_split', 'min_samples_leaf',
                              'max_depth', 'max_leaf_nodes', 'min_impurity_decrease', 'max_features']

        return clf, parameter_space, possible_param

    def fit(self, model: str, X: Iterable, y: Iterable, num_images_person: int):
        if model == 'knn':
            self.model = neighbors.KNeighborsClassifier(algorithm='brute',
                                                        # n_neighbors=max(1, int(num_images_person * 3 / 4)),
                                                        n_neighbors=max(1, int(num_images_person-1)),
                                                        weights='distance', p=2)
        elif model == "dtree":
            self.model = DecisionTreeClassifier(criterion='entropy', max_depth=140, random_state=self.rand_seed)
        elif model == 'random_forest':
            self.model = ensemble.RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=140,
                                                         random_state=self.rand_seed)
        elif model == 'mlp':
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

        elif model == 'svm':
            # self.model = svm.SVC(kernel='poly', degree=2, C=1e-6, gamma=3.33e-5, tol=1e-7, probability=True,
            #                      random_state=self.rand_seed)
            self.model = svm.SVC(kernel='poly', degree=2, C=1e+6, gamma="auto", tol=1e-7, probability=True,
                                 random_state=self.rand_seed)
        else:  # svm
            self.model = svm.SVC(kernel='poly', degree=2, C=1e-6, gamma=3.33e-5, tol=1e-7, probability=True,
                                 random_state=self.rand_seed)

        start_time = time()
        self.model.fit(X, y)
        self.fit_time = time() - start_time

    def train(self, X: Iterable, y: Iterable, model_name: str = 'knn', num_sets: Optional[Num_Type] = 10,
              images_per_person: Union[int, Tuple[int, int]] = 10, num_people: Optional[int] = None,
              test_images_id: Optional[Iterable] = None, X_test: Optional[Iterable] = None,
              y_test: Optional[Iterable] = None):

        global output_folder, temp_logger_path, data_folder

        data_folder = path.join(output_folder, model_name)
        makedirs(data_folder, exist_ok=True)

        score: Optional[float] = None
        scoring_report: Optional[dict] = None
        prob_report: Optional[list] = None
        train_test_images: Optional[tuple] = None

        if self.parameter_tuning is True:
            cv = StratifiedKFold(n_splits=num_sets, shuffle=True, random_state=self.rand_seed)
            score = self.tuning(model_name, cv, X, y)

        # elif k_fold is True:
        #     raise NotImplementedError
        #     cv = StratifiedKFold(n_splits=num_sets)
        #     self.choose_model()
        #     # selecionar melhores metricas, e armazena-las em csv separado, com os respectivos parametros usados
        #     scoring = {'balanced_accuracy_score': 'balanced_accuracy_score',
        #                'precision_weighted': 'precision_weighted',
        #                'f1_weighted': 'f1_weighted,',
        #                'recall_weighted': 'recall_weighted',
        #                'roc_auc_score': 'roc_auc_score',
        #                'weighted_roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
        #                'fit_time': 'fit_time',
        #                'score_time': 'score_time'
        #                }
        #     score = cross_validate(self.model, X, y, cv, scoring=scoring)
        #     print(f"Scores:\n{score}")

        else:
            if type(images_per_person) == tuple:
                images_train, images_test = images_per_person
                train_test_images = images_per_person
                images_per_person = images_train + images_test

            else:
                if num_sets > 1:
                    # images_train_person = images_per_person
                    X, X_test, y, y_test = train_test_split(X, y, stratify=y,
                                                            test_size=1 / num_sets,
                                                            random_state=self.rand_seed)

                images_train = (int((num_sets - 1) * images_per_person / num_sets) if num_sets > 1
                                else (num_sets * images_per_person / num_sets))

                train_test_images = (images_train, images_per_person - images_train)

            self.fit(model_name, X, y, num_images_person=images_train)

            if X_test is not None and y_test is not None:
                start_time = time()
                y_pred = self.model.predict(X_test)
                self.score_time = time() - start_time

                y_prob = self.model.predict_proba(X_test)
                chosen = np.argmax(y_prob, axis=1)

                # print(y_prob[0:8])
                # print(chosen)

                prob_report = [[None] for _ in range(y_pred.size + 1)]
                prob_report[0] = ["Class", "Image_Num", "Prediction", "Probability"]

                for i in range(len(y_pred)):
                    prob_report[i + 1] = [y_test[i], test_images_id[i], y_pred[i], y_prob[i][chosen[i]]]


                scoring_report = classification_report(y_test, y_pred, output_dict=True)

                score = f1_score(y_test, y_pred, average='weighted')

        if output_folder is not None and score is not None:
            self.save_data(images_per_person, model_name, num_people, num_sets, score,
                           scoring_report, prob_report, train_test_images)

        try:
            remove(temp_logger_path)
        except (OSError, TypeError):
            pass

        return score

    def save_data(self, images_per_person: int, model_name: str, num_people: int, num_sets: Optional[Num_Type],
                  score: float, scoring_report: Optional[dict] = None, prob_report: Optional[list] = None,
                  train_test_images: Optional[Tuple[int, int]] = None):
        global data_folder, output_folder

        try:
            with open(self.get_model_path(model_name), 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as err:
            print(f"Nao foi possivel salvar modelo: {err}")

        file_id = self.save_parameters(images_per_person, model_name, num_people, num_sets, score)

        if scoring_report is not None:
            score_f = path.join(data_folder, f"{model_name}_scoring_{file_id}")
            prob_f = path.join(data_folder, f"{model_name}_probability_{file_id}")

            import json, csv
            # with open(f"{score_f}.json", 'w') as f:
            #     json_report = json.dumps(scoring_report)
            #     f.write(json_report)

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
                        # f.write(f"{';'.join(to_str(v) for v in row)}\n")

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

    def save_parameters(self, images_per_person: int, model_name: str, num_people: int,
                        num_sets: int, score: float) -> Union[int, str]:
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

    def classify(self, embedding: Iterable):
        if self.model is None:
            print('Train the model before doing classifications.')
            return

        pred = self.model.predict([embedding])
        # if len(pred) > 1:
        #     print("Houston, we have a problem")

        # Para knn, a probabilidade so deve ser diferente de 1 para maiores valores de k,
        # mas melhor reconhecimento tem ocorrido com k=1
        prob = np.ravel(self.model.predict_proba([embedding]))

        # selecionar maiores probabilidades para 2o classificador
        # prob[::-1].sort()

        return pred[0], round(np.amax(prob), 2)
