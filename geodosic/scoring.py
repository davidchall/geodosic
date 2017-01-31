# third-party imports
import numpy as np
from sklearn.metrics import r2_score


def generate_dvhs(model, X, n_dose_bins=100, **kwargs):
    for p in X:
        if model.dose_name not in p.dose_names:
            continue
        if model.target_name not in p.structure_names:
            continue

        for oar_name in model.oar_names:
            if oar_name not in p.structure_names:
                continue

            # choose appropriate binning for DVHs
            dose = p.dose_array(model.dose_name, model.dose_name)
            target_mask = p.structure_mask(model.target_name, model.dose_name)
            max_target_dose_plan = np.max(dose[target_mask])

            if model.normalize_to_prescribed_dose:
                max_target_dose_pred = p.prescribed_doses[model.dose_name]
            else:
                max_target_dose_pred = model.max_prescribed_dose

            max_dvh_dose = max(max_target_dose_plan, max_target_dose_pred)
            dose_edges = np.linspace(0, 1.2*max_dvh_dose, n_dose_bins)

            dvh_pred = model.predict_structure(p, oar_name, dose_edges=dose_edges, **kwargs)
            dvh_plan = p.calculate_dvh(oar_name, model.dose_name, dose_edges=dose_edges)

            yield p, dvh_pred, dvh_plan


def score_dvh_metric(model, X, y=None,
                     dvh_metric=lambda dvh: dvh.mean(),
                     **kwargs):
    """Scores the model by comparing a metric of the predicted and actual DVH.

    Args:
        X: cohort of Patient objects
        dvh_metric: function that converts DVH into a metric
    """
    y_true, y_pred = [], []
    for p, dvh_pred, dvh_plan in generate_dvhs(model, X, **kwargs):
        y_pred.append(dvh_metric(dvh_pred))
        y_true.append(dvh_metric(dvh_plan))
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    return r2_score(y_true, y_pred)
