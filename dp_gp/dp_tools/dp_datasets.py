import numpy as np
import pandas as pd
from dp_gp.dp_tools.mechanisms import laplace_mechanism, gauss_mechanism
from sklearn.model_selection import train_test_split


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
    test_size: float = 0.2,
):
    """
    Load and prepare CitiBike dataset, optionally normalize and clip features and targets
    to above pre-defined values then apply local differential privacy.
    Args:
        normalize (bool): whether to normalize the data
        clip_features (bool): whether to clip feature values to pre-defined ranges (for differential privacy)
        clip_targets (bool): whether to clip target values to pre-defined ranges (for differential privacy)
        mechanism (str): privacy mechanism to use, one of "lap" or "gauss"
        eps (float): privacy budget epsilon
        delta (float): failure probability delta (set to 0.0 when using Laplace noise)
        test_size (float): relative validation set size 
    Returns:
        X, y: feature and target arrays
    """
    data = pd.read_csv("/home/moritz/repositories/thesis/data/citibike_tripdata_201306.csv", index_col="Unnamed: 0")
    X = np.array(data.drop("tripduration", axis=1))
    y = np.array(data["tripduration"])

    y = np.expand_dims(y, 1)
    assert np.sum(np.isnan(X)) + np.sum(np.isnan(y)) == 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    mech = laplace_mechanism if mechanism == "lap" else gauss_mechanism

    # clip, then privatise targets y
    if private_targets:
        y_train = mech(
            inp_arr=np.clip(y, a_min=MIN_TRIP_DURATION, a_max=MAX_TRIP_DURATION),
            eps=eps,
            delta=delta,
            sens=MAX_TRIP_DURATION - MIN_TRIP_DURATION,
        )
    # clip, then privatise features X
    if private_features:
        lats_start = mech(
            inp_arr=np.clip(
                X_train[:, 0], a_min=MIN_LATITUDE, a_max=MIN_LATITUDE + CLIP_X
            ),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        longs_start = mech(
            inp_arr=np.clip(
                X_train[:, 1], a_min=MIN_LONGITUDE, a_max=MIN_LONGITUDE + CLIP_X
            ),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        lats_end = mech(
            inp_arr=np.clip(
                X_train[:, 2], a_min=MIN_LATITUDE, a_max=MIN_LATITUDE + CLIP_X
            ),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        longs_end = mech(
            inp_arr=np.clip(
                X_train[:, 3], a_min=MIN_LONGITUDE, a_max=MIN_LONGITUDE + CLIP_X
            ),
            eps=eps,
            delta=delta,
            sens=CLIP_X,
        )
        X_train = np.stack([lats_start, longs_start, lats_end, longs_end], axis=-1)
    if normalize:
        raise NotImplementedError()
    return (X_train, y_train), (X_test, y_test)



MIN_AGE = 0
MAX_AGE = 100
MIN_HEIGHT = 50
MAX_HEIGHT = 180

def load_kung_sa_dataset(
    normalize: bool = False,
    private_features: bool = False,
    private_targets: bool = False,
    mechanism="lap",
    eps: float = 1.0,
    delta: float = 0.0,
    test_size:float=0.2,
):
    """
    Load and prepare Kung Sa! Women dataset, optionally normalize and clip features
    and targets to above pre-defined values, then apply local differential privacy.
    Args:
        normalize (bool): whether to normalize the data
        clip_features (bool): whether to clip feature values to pre-defined ranges (for differential privacy)
        clip_targets (bool): whether to clip target values to pre-defined ranges (for differential privacy)
        mechanism (str): privacy mechanism to use, one of "lap" or "gauss"
        eps (float): privacy budget epsilon
        delta (float): failure probability delta (set to 0.0 when using Laplace noise)
        test_size (float): relative validation set size 
    Returns:
        X, y: feature and target arrays
    """

    howell = pd.read_csv("/home/moritz/repositories/thesis/data/Howell1.csv", sep=";")
    howell_female = howell[howell["male"] == 0]
    X, y = np.array(howell_female["age"]), np.array(howell_female["height"])
    X, y = np.expand_dims(X, 1), np.expand_dims(y, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    mech = laplace_mechanism if mechanism == "lap" else gauss_mechanism

    if private_features:
        X_train = mech(
            np.clip(X_train, a_min=MIN_AGE, a_max=MAX_AGE),
            eps=eps,
            delta=delta,
            sens=MAX_AGE - MIN_AGE,
        )
    if private_targets:
        y_train = mech(
            np.clip(y_train, a_min=MIN_HEIGHT, a_max=MAX_HEIGHT),
            eps=eps,
            delta=delta,
            sens=MAX_HEIGHT - MIN_HEIGHT,
        )
    if normalize:
        raise NotImplementedError()
    return (X_train, y_train), (X_test, y_test)


