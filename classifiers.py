import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Implementacao do classificador de K vizinhos mais proximos (K Nearest Neighbors)
class KNN_Classifier():
    def __init__(self, X_train, y_train, K = 3):        
        self.x, self.y = X_train.values, y_train.values
        self.K = K

        self.indexes = np.arange(len(y_train))

    # distancia euclidiana entre duas linhas
    def distance(self, x1, x2):
        # d = 0
        # for i in range(len(x1)):
        #     d += (x1[i] - x2[i])**2

        # return np.sqrt(d)

        # numpy para rodar mais rapido
        return np.sqrt( np.sum( (x1-x2)**2 ) )

    # retorna os indices dos K vizinhos mais proximos
    def nearest_neighbors(self, x):
        distances = np.empty((len(self.indexes), 2))

        # calcula distancias da linha a ser testada para
        # cada uma das linhas de teste
        for i in self.indexes:
            train_x = self.x[i]
            dist = self.distance(x, train_x)
            distances[i] = [i, dist]

        sorted = distances[ distances[:,1].argsort() ]

        # retorna os K com menor distancia
        nn = np.array([sorted[i][0] for i in range(self.K)])
        
        return nn.astype(int)

    # retorna o y esperado para cada linha de X_test
    def predict(self, X_test):
        count = len(X_test.values)
        result = np.empty(count)

        for i in range(count):
            nn_indexes = self.nearest_neighbors(X_test.values[i])
            y_result = self.y[nn_indexes]
            result[i] = np.argmax(np.bincount(y_result)) # pega o y mais frequente dos K vizinhos

        return result

# Implementacao do classificador de florestas aleatorias (Random Forests)
# Usa o BaseEstimator e ClassifierMixin para ser compativel com o RandomizedSearchCV
class RF_Classifier(BaseEstimator, ClassifierMixin):
    def __init__ (self, n_estimators=100, max_features='auto', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        # parametros da rf
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.trees = []

    def fit(self, X_train, y_train):
        self.x, self.y = X_train.values, y_train.values
        self.n = len(y_train)

        # cria as n arvores da floresta
        for i in range(self.n_estimators):
            self.trees.append(self.create_tree())
    
    def create_tree(self):
        # gera as N amostras aleatorias, com reposicao
        rng = np.random.default_rng()
        random_indexes = rng.choice(self.n, replace=True, size=self.n)
        
        # seleciona na ordem gerada, gera decision tree com essas amostras
        random_samples = self.x[random_indexes]
        rs_labels = self.y[random_indexes]

        # usa decision tree do sklearn, com os parametros definidos
        return DecisionTreeClassifier(max_features=self.max_features, 
                                      max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf).fit(random_samples, rs_labels)
    
    def predict(self, x):
        # media das previsoes de todas as arvores geradas
        tree_values = [ t.predict(x.values) for t in self.trees ]
        results = np.mean(tree_values, axis=0)
        
        return [ (1 if r > 0.5 else 0) for r in results]