import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_squared_log_error,
                             f1_score)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def verify_numeric_prediction(y_true, y_pred):
    obs_1 = mean_absolute_error(y_true, y_pred)
    obs_2 = mean_squared_error(y_true, y_pred)
    obs_3 = mean_squared_log_error(y_true, y_pred)

    obs = [obs_1, obs_2, obs_3]

    for o in obs:
        assert isinstance(o, (int, float))
        assert 0 <= o

    return


def verify_nominal_prediction(y_true, y_pred):
    obs = f1_score(y_true, y_pred, average='macro')

    assert isinstance(obs, (int, float))
    assert 0 <= obs <= 1

    return


