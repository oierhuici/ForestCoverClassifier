#Imports de librerias y carga de datos

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest

#Importamos los datos de entrenamiento y test
data_train = pd.read_csv("../input/data_train.csv").values
data_test = pd.read_csv("../input/data_test.csv").values

#Definicion de la matriz de costes
C = [[ 0,   5,  1, 1, 1,  1, 1],
[10,   0,  1, 1, 1,  1, 1],
[20,  20,  0, 5, 5, 50, 5],
[20,  20, 10, 0, 1, 50, 5],
[20, 100,  5, 1, 0,  5, 5],
[ 5,  10, 10, 5, 1,  0, 1],
[10,   5,  1, 1, 1,  1, 0]]


#Preprocesamiento: Eliminar outliers

#Modelo Isolation Forest
model = IsolationForest(contamination="auto", random_state=30)
model.fit(data_train)

# Etiquetas de outliers (1 para inliers, -1 para outliers) => Pasar a True para inliers y False para outliers
outlier_labels = model.predict(data_train)
outlier_labels = np.asarray(np.where(outlier_labels==-1, 0, outlier_labels)).T
outlier_labels = [bool(outlier_labels[i]) for i in range(len(outlier_labels))]

#Filtramos el dataset para quedarnos solo con inliers
data_train_inliers = data_train[outlier_labels]
data_train = data_train_inliers


#Entrenamiento y prediccion

# Configuración del clasificador XGBoost
clf = xgb.XGBClassifier(
                        random_state = 30,
                        n_estimators =  200,
                        learning_rate = 0.1,
                        max_depth = 30,
                        objective = 'multi:softmax',
                        num_class = 7,
                        verbosity = 2,
                        subsample = 0.8,
                        colsample_bytree= 1.0,
                        min_split_loss = 0
                        )

# Entrenamiento del modelo
clf.fit(data_train[:,0:54], data_train[:,54])

# Obtenemos la probabilidad de cada muestra de pertenecer a una clase
y_proba = clf.predict_proba(data_test)

#Calculamos los pesos ponderados de acuerdo con la matriz de costes
weighted_costs = np.matmul(y_proba, np.array(C))

#Escogemos la clase como el peso minimo
y_pred = np.argmin(weighted_costs, axis = 1)

#Generacion del archivo
id = np.arange(data_test.shape[0])
y = pd.DataFrame(
     {
          "Id": id,
          "Category": y_pred,
     }
)
nombre = "solution.csv"
y.to_csv(nombre, index=False)