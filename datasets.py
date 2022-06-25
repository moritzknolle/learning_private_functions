import numpy as np
import pandas as pd
from dp_gp.dp_tools.mechanisms import laplace_mechanism, gauss_mechanism

MAX_TRIP_DURATION = 2_000
MIN_TRIP_DURATION = 0

MIN_LATITUDE = 40.680
MIN_LONGITUDE = -74.02
CLIP_X = 0.09


def load_citibike_data(
    normalize: bool = False,
    private_features: bool = False,
    private_targets: bool = False,
    mechanism="lap",
    eps: float = 1.0,
    delta: float = 0.0,
):
    """
    Load and prepare CitiBike dataset, optionally normalize and clip features and targets to above pre-defined values.
    Args:
        normalize (bool): whether to normalize the data
        clip_features (bool): whether to clip feature values to pre-defined ranges (for differential privacy)
        clip_targets (bool): whether to clip target values to pre-defined ranges (for differential privacy)
        mechanism (str): privacy mechanism to use, one of "lap" or "gauss"
        eps (float): privacy budget epsilon
        delta (float): failure probability delta (set to 0.0 when using Laplace noise)
    Returns:
        X, y: feature and target arrays
    """
    data = pd.read_csv("data/citibike_tripdata_201306.csv", index_col="Unnamed: 0")
    X = np.array(data.drop("tripduration", axis=1))
    y = np.array(data["tripduration"])
    assert np.sum(np.isnan(X)) + np.sum(np.isnan(y)) == 0

    mech = laplace_mechanism if mechanism == "lap" else gauss_mechanism

    # clip then privatise targets
    if private_targets:
        y_priv = mech(
            inp_arr=np.clip(y, a_min=MIN_TRIP_DURATION, a_max=MAX_TRIP_DURATION),
            eps=eps,
            delta=delta,
            sens=MAX_TRIP_DURATION - MIN_TRIP_DURATION,
        )
        y = y_priv
        # y_diff = y_clipped - y
        # print(f"Clip coverage: targets={(1-np.count_nonzero(y_diff)/y.size)*100}%")

    # clip then privatise featuers
    if private_features:
        lats_start = mech(
            inp_arr=np.clip(X[:, 0], a_min=MIN_LATITUDE, a_max=MIN_LATITUDE + CLIP_X),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        longs_start = mech(
            inp_arr=np.clip(X[:, 1], a_min=MIN_LONGITUDE, a_max=MIN_LONGITUDE + CLIP_X),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        lats_end = mech(
            inp_arr=np.clip(X[:, 2], a_min=MIN_LATITUDE, a_max=MIN_LATITUDE + CLIP_X),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        longs_end = mech(
            inp_arr=np.clip(X[:, 3], a_min=MIN_LONGITUDE, a_max=MIN_LONGITUDE + CLIP_X),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        X_priv = np.stack([lats_start, longs_start, lats_end, longs_end], axis=-1)
        X = X_priv
    if normalize:
        pass
    return X, y
