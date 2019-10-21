import pickle
from os import path, makedirs
import numpy as np
from sklearn import neighbors, svm, ensemble, neural_network
from sklearn.metrics import make_scorer, classification_report, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

BASE_DIR = path.dirname(__file__)
PATH_TO_PKL = 'trained_classifier.pkl'


class FaceClassifier:
    def __init__(self, model_path=None):

        self.model = None
        if model_path is None:
            return
        elif model_path == 'default':
            model_path = path.join(BASE_DIR, PATH_TO_PKL)

        # Load models
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def parameter_tuning(self, model, cv, images_per_person, X, y):
        print("Parameter tuning")
        if model == "svm":
            parameter_space = {
                'C': (0.001, 1000000.0),
                'gamma': (0.00001, 1.0),
                'degree': (1, 9),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'tol': (0.00001, 0.1)
            }
            clf = svm.SVC(probability=True)

        elif model == "knn":
            parameter_space = {
                'n_neighbors': (1, 2 * images_per_person),
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': (5, 150),
                'p': (1, 9)
            }
            clf = neighbors.KNeighborsClassifier()

        elif model == "mlp":
            parameter_space = {
                # 'layer1': Integer(5, 100),
                # 'layer2': Integer(0, 100),
                # 'layer3': Integer(0, 100),
                'hidden_layer_sizes': Integer(10, 150),
                # numpy.arange(0.005, 0.1, 0.005)
                'activation': Categorical(['relu', 'tanh', 'logistic']),
                'solver': Categorical(['adam', 'sgd', 'lbfgs']),
                'alpha': Real(0.01, 0.5),
                'learning_rate': Categorical(['constant', 'adaptive']),
                'learning_rate_init': Real(0.002, 0.2),
                'max_iter': Integer(9999, 10001)
            }
            clf = neural_network.MLPClassifier()

        elif model == "dtree":
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
        score = make_scorer(f1_score, average='weighted')
        clf = BayesSearchCV(estimator=clf, search_spaces=parameter_space, n_iter=50, cv=cv,
                            scoring=score, verbose=True, n_jobs=-1)
        clf.fit(X, y)
        self.model = clf.best_estimator_
        return clf.best_score_

    def choose_model(self, model):
        if model == 'knn':
            self.model = neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', p=2, leaf_size=110)
        elif model == "dtree":
            self.model = DecisionTreeClassifier(criterion='entropy', max_depth=140)
        elif model == 'random_forest':
            self.model = ensemble.RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=140)
        elif model == 'mlp':
            self.model = neural_network.MLPClassifier(activation='tanh', hidden_layer_sizes=60,
                                                      learning_rate='adaptive', learning_rate_init=0.08,
                                                      max_iter=10000, alpha=0.093, solver='adam')
        elif model == 'svm':
            self.model = svm.SVC(kernel='rbf', C=10000.0, probability=True)
        else:  # svm
            self.model = svm.SVC(kernel='linear', probability=True)

    def train(self, X, y, model='knn', num_sets=10, k_fold=False, parameter_tuning=False, save_model_path=None,
              images_per_person=10, num_people=None):

        if parameter_tuning is True:
            cv = StratifiedKFold(n_splits=num_sets, shuffle=True)
            score = self.parameter_tuning(model, cv, images_per_person, X, y)

        elif k_fold is True:
            raise NotImplementedError
            cv = StratifiedKFold(n_splits=num_sets, shuffle=True)
            self.choose_model()
            # TODO: selecionar melhores metricas, e armazena-las em csv separado, com os respectivos parametros usados
            scoring = {'balanced_accuracy_score': 'balanced_accuracy_score',
                       'precision_weighted': 'precision_weighted',
                       'f1_weighted': 'f1_weighted,',
                       'recall_weighted': 'recall_weighted',
                       'roc_auc_score': 'roc_auc_score',
                       'weighted_roc_auc_score': make_scorer(roc_auc_score, average='weighted'),
                       'fit_time': 'fit_time',
                       'score_time': 'score_time'
                       }
            score = cross_validate(self.model, X, y, cv, scoring=scoring)
            print(f"Scores:\n{score}")

        else:
            if num_sets > 1:
                X, X_test, y, y_test = train_test_split(X, y, stratify=y, test_size=1 / num_sets)

            self.choose_model()

            self.model.fit(X, y)

            if num_sets > 1:
                y_pred = self.model.predict(X_test)
                # y_prob = self.model.predict_proba(X_test)
                print(classification_report(y_test, y_pred))
                score = f1_score(y_test, y_pred, average='weighted')

        if save_model_path is not None:
            folder = path.dirname(save_model_path)
            makedirs(folder, exist_ok=True)

            with open(save_model_path, 'wb') as f:
                pickle.dump(self.model, f)

            parameters_csv = path.join(path.dirname(save_model_path), f"{model}_parameters.csv")
            parameters = self.model.get_params()
            if not path.exists(parameters_csv):
                file = open(parameters_csv, "w")
                file.write(';'.join(param_name for param_name in parameters.keys()) +
                           ";images_per_person;number_people;number_sets;score\n")
            else:
                file = open(parameters_csv, 'a')
            file.write(';'.join(str(param_value) for param_value in list(parameters.values())) +
                       f";{images_per_person};{num_people};{num_sets};{score}\n")
            file.close()

    def classify(self, descriptor):
        if self.model is None:
            print('Train the model before doing classifications.')
            return

        pred = self.model.predict([descriptor])
        # if len(pred) > 1:
        #     print("Houston, we have a problem")

        # Para knn, a probabilidade so deve ser diferente de 1 para maiores valores de k,
        # mas melhor reconhecimento tem ocorrido com k=1
        prob = np.ravel(self.model.predict_proba([descriptor]))
        # selecionar maiores probabilidades para 2o classificador
        # prob[::-1].sort()

        return pred[0], round(np.amax(prob), 2)
