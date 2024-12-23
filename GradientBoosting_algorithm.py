### Código para hacer pruebas de complejidad con Gradient Boosting
### Vamos a cambiar los pesos iniciales
### y los pesos del fit

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Tengo que hacer:
# cambios con el init: fácil y difícil
# cambios con el sample_weights: fácil y difícil
# combo de cambios
# hay 2 funciones pérdida log-loss y exponential (que es adaboost)


## Con esta función puedo modificar los pesos iniciales

class CustomInitModel(BaseEstimator, ClassifierMixin):
    def __init__(self, initial_weights):
        self.initial_weights = initial_weights

    def fit(self, X, y):
        # No se necesita ajuste, ya que los pesos están predefinidos
        return self

    def predict_proba(self, X):
        # Devuelve probabilidades basadas en los pesos iniciales
        proba = np.tile(self.initial_weights, (X.shape[0], 1))
        return proba

# Vector de pesos iniciales para dos clases
initial_weights = np.array([0.7, 0.3])  # Clase 0: 70%, Clase 1: 30%

# Usar el modelo inicial personalizado
init_model = CustomInitModel(initial_weights=initial_weights)

# Aplicarlo en GradientBoostingClassifier
gbc = GradientBoostingClassifier(init=init_model, n_estimators=100, random_state=42)
gbc.fit(X_train, y_train)

