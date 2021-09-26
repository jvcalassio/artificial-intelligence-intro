import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics

from classifiers import KNN_Classifier, RF_Classifier


def process_file(df):
    # adapta strings para numericos
    df = df.replace({'negative': 0, 'positive': 1, # resultado do exame
                        'not_detected': 0, 'detected': 1,
                        'not_done': np.NaN, 'Não Realizado': np.NaN,
                        '<1000': np.NaN,
                        'absent': 0, 'present': 1,
                        'clear': 0, 'lightly_cloudy': 1, # Urine - Aspect
                        'cloudy': 2, 'altered_coloring': 3,
                        'normal': 0, # Urine - Urobilinogen
                        'Ausentes': 0, 'Oxalato de Cálcio -++': 1, # Urine - Crystals
                        'Oxalato de Cálcio +++': 2, 'Urato Amorfo --+': 3,
                        'Urato Amorfo +++': 4, 
                        'light_yellow': 0, 'yellow': 1, # Urine - Color
                        'citrus_yellow': 2, 'orange': 3,
                        }) 
    # resolve casos especificos do arquivo
    # 1. numericos com "" (ex: "6,5") que nao sao lidos como float
    # 2. numerico que nao eh lido como float pois nao tem separador
    df['Urine - pH'] = df['Urine - pH'].astype('str').apply(lambda x: x.replace(',', '.')).astype('float')
    df['Urine - Leukocytes'] = df['Urine - Leukocytes'].astype('float')

    # remove as colunas em branco
    df = df.dropna(axis='columns', how='all')

    y = df['resultado do exame'] # classes sao o resultado do exame
    X = df.drop(columns=['resultado do exame']) # colunas a serem consideradas

    # substitui valores vazios (NaN) nas colunas por 0
    X[:] = np.nan_to_num(X)

    return (X, y)

# otimizacao dos parametros da random forest
def adjust_parameters(X_train, y_train):
    # parametros ajustados:
    # n_estimators = numero de arvores da floresta
    # max_features = numero maximo de caracteristicas consideradas ao dividir
    # max_depth = max. de niveis em cada arvore
    # min_samples_split = numero minimmo de dados em um no antes de ser dividido
    # min_samples_leaf = numero minimo de dados permitos num no folha

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_depth = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
    max_depth.append(None)

    # define o grid de hiperparametros
    random_grid = { 'n_estimators': n_estimators,
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': max_depth,
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4] }
    
    clf = RF_Classifier()

    # realiza a busca aleatoria de quais as melhores combinacoes de parametros 
    # utiliza a RandomizedSearchCV da sklearn para isso, com 100 iteracoes
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, n_jobs=-1)
    rf_random.fit(X_train, y_train)

    print("Melhores parametros: ", rf_random.best_params_, "\n")

    # escolhe e retorna o melhor estimador (com os melhores parametros)
    best_random = rf_random.best_estimator_

    return best_random

def print_metrics(y_test, y_pred):
    print("\nEstatisticas: ")
    print("Acuracia (exatidao):", metrics.accuracy_score(y_test,y_pred))
    print("Precisao:", metrics.precision_score(y_test,y_pred,pos_label=0))
    print("Recuperacao:", metrics.recall_score(y_test,y_pred,pos_label=0))
    print("Matriz de confusao:\n", metrics.confusion_matrix(y_test,y_pred))

def main():
    
    print("Escolha um classificador (1 ou 2):")
    print("1. Random Forests")
    print("2. K-Nearest Neighbors")

    option = int(input())
    if option == 1: # random forest
        optimize = int(input("Otimizar os parametros da Random Forest? 1 = Sim, 0 = Nao\n"))
    
    # leitura do arquivo
    df = pd.read_excel('dataset.xlsx')

    # pre-processamento
    X, y = process_file(df)

    # separa conjunto de treinamento e de teste : 75% treinamento / 25% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

    if option == 1:
        if optimize == 1:
            print("Otimizando parametros...")
            clf = adjust_parameters(X_train, y_train)
        else:
            clf = RF_Classifier()
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
    elif option == 2:
        clf = KNN_Classifier(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(y_pred)
    else:
        exit()

    # mostra metricas do classificador
    print_metrics(y_test, y_pred)

main()