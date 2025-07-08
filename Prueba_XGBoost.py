# Si queremos hacerlo con xgboost, podemos usar una función de pérdida customizada
# pero también necesitamos la hessiana


import numpy as np
import xgboost as xgb

# Definimos nuestra propia función de pérdida
def complexity_weighted_logloss(complexity_array):
    def custom_logloss(preds, dtrain):
        y_true = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Convert logits to probabilities

        # Compute weighted gradients and Hessians
        grad = complexity_array * (preds - y_true)
        hess = complexity_array * preds * (1 - preds)

        return grad, hess
    return custom_logloss

# Train XGBoost model with custom loss



dtrain = xgb.DMatrix(X_train, label=y_train, weight=complexity_array)

params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train using your custom objective
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=complexity_weighted_logloss(complexity_array)
)

# to evaluate your model with the same complexity-aware loss, you can define a custom evaluation function

def complexity_weighted_logloss_eval(complexity_array):
    def eval_metric(preds, dtrain):
        y_true = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))

        eps = 1e-15  # To avoid log(0)
        logloss = -np.sum(
            complexity_array * (
                y_true * np.log(preds + eps) +
                (1 - y_true) * np.log(1 - preds + eps)
            )
        ) / np.sum(complexity_array)

        return 'complexity_logloss', logloss
    return eval_metric


bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=complexity_weighted_logloss(complexity_array),
    feval=complexity_weighted_logloss_eval(complexity_array),
    evals=[(dtrain, 'train')],
    verbose_eval=True
)


