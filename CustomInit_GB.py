
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from All_measures import all_measures
#import random # for sampling with weights
from sklearn import preprocessing
#from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import log_loss




# Con esta función puedo modificar las predicciones iniciales

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomInitModel(BaseEstimator, ClassifierMixin):
    #     # La clase CustomInitModel hereda de dos clases base de scikit-learn: BaseEstimator y ClassifierMixin.
    #     # Esto hace que la clase CustomInitModel tenga un comportamiento compatible con scikit-learn.
    """
    Este modelo personalizado permite inicializar GradientBoostingClassifier con predicciones F0 específicas.
    Funciona para funciones de pérdida como log_loss y exponential, que trabajan sobre logits.
    """

    def __init__(self, initial_predictions=None, loss='log_loss'):
        """
        Parámetros:
        - initial_predictions: array de logits iniciales (F0), uno por cada muestra del conjunto de entrenamiento.
        - loss: 'log_loss' o 'exponential' (ambas usan logits como entrada).
        """
        self.initial_predictions = initial_predictions
        self.loss = loss

    def fit(self, X, y):
        """
        En entrenamiento:
        - Usa las predicciones iniciales si se proporcionan.
        - También calcula el logit del promedio de y (F0 constante por defecto de sklearn) para usar en test.
        """
        y = np.asarray(y)

        # Si no se dan predicciones iniciales, usamos un vector de ceros (equivale a prob 0.5 → logit(0.5) = 0)
        if self.initial_predictions is None:
            self.initial_predictions_ = np.zeros_like(y, dtype=float)
        else:
            if len(self.initial_predictions) != len(y):
                raise ValueError("initial_predictions must match number of training samples")
            self.initial_predictions_ = np.array(self.initial_predictions)

        # F₀ por defecto que usaría sklearn: logit(p) con p = promedio de y_train
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)  # Evita log(0)
        self.default_logit_ = np.log(p / (1 - p))

        return self

    def predict_proba(self, X):
        """
        Devuelve las probabilidades predichas a partir de los logits.
        - En train: usa las predicciones personalizadas.
        - En test: usa F₀ constante igual al de sklearn.
        """
        n = X.shape[0]

        if n == len(self.initial_predictions_):
            # Train: usar las predicciones personalizadas
            pred = self.initial_predictions_
        else:
            # Test: usar logit constante (F₀) basado en el promedio de y_train
            pred = np.full(n, self.default_logit_)

        # Convertir los logits a probabilidades: σ(x) = 1 / (1 + exp(-x))
        probas = 1.0 / (1.0 + np.exp(-pred))

        # Evitar probabilidades exactas de 0 o 1 (por estabilidad numérica)
        probas = np.clip(probas, 1e-6, 1 - 1e-6)

        # Devolver en formato [P(0), P(1)] por fila
        return np.vstack([1 - probas, probas]).T

    def predict(self, X):
        """
        Predicción final de clases: aplica un umbral de 0.5 sobre P(1).
        """
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# 1. Crear datos de clasificación binaria
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 2. Crear una "complejidad" falsa: puntaje aleatorio entre 0 y 1
np.random.seed(42)
complexity_score = np.random.rand(len(y_train))  # Simulando medida de complejidad

# 3. Escalarlo entre 0.05 y 0.95 para evitar extremos 0 o 1
scaled_complexity = 0.05 + 0.9 * complexity_score

init_logits = np.log((scaled_complexity + 1e-5) / (1 - scaled_complexity + 1e-5))
init_model = CustomInitModel(initial_predictions=init_logits)

clf = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=1,
    init=init_model
)
clf.fit(X_train, y_train)

# 7. Evaluar
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba))
