# third-party imports
import numpy as np
from sklearn.metrics import r2_score


def score_dvh_metric(model, X, y=None,
                     dvh_metric=lambda dvh: dvh.mean(),
                     score_metric=r2_score,
                     **kwargs):
    """Scores the model by comparing a metric of the predicted and actual DVH.

    Args:
        model: estimator
        X: cohort of Patient objects
        dvh_metric: function that converts DVH into a metric
        score_metric:

    Returns:
        score: floating point number that quantifies the prediction quality
    """
    y_true, y_pred = [], []
    for _, dvh_pred, dvh_plan in model.generate_validation_dvhs(X, **kwargs):
        y_pred.append(dvh_metric(dvh_pred))
        y_true.append(dvh_metric(dvh_plan))
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return score_metric(y_true, y_pred)
