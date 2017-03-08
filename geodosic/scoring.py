# standard imports
import warnings

# third-party imports
import numpy as np
from sklearn.metrics import r2_score
from npgamma import calc_gamma


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


def score_gamma_index(model, X, y=None,
                      dose_threshold=3, distance_threshold=3,
                      return_raw=False,
                      n_jobs=2,
                      **kwargs):
    """Scores the model by comparing the 3D dose distributions of the predicted
    and actual treatment plans using the gamma index.

    Args:
        model: estimator
        X: cohort of Patient objects
        dose_threshold: [%] used in gamma index
        distance_threshold: [mm] used in gamma index
        return_raw: return array of gamma indices
        n_jobs: number of threads
        kwargs: passed to model.generate_validation_dose_arrays

    Returns:
        score: negative mean square error from pass rate of unity
    """
    pass_rates = []
    for p, grid, dose_pred, dose_plan in model.generate_validation_dose_arrays(X, **kwargs):

        if np.all(dose_plan == 0):
            warnings.warn('Skipping patient with D_ref=0', RuntimeWarning)
            continue

        abs_dose_threshold = dose_threshold / 100. * dose_plan.max()

        gamma = calc_gamma(
            grid, dose_plan,
            grid, dose_pred,
            distance_threshold, abs_dose_threshold,
            lower_dose_cutoff=1e-12,
            maximum_test_distance=2*distance_threshold,
            num_threads=n_jobs)

        valid_gamma = gamma[~np.isnan(gamma)]
        pass_rate = np.sum(valid_gamma <= 1) / len(valid_gamma)

        pass_rates.append(pass_rate)

    pass_rates = np.array(pass_rates)

    if return_raw:
        return pass_rates

    mse = np.mean(np.square(1. - pass_rates))
    return -mse
