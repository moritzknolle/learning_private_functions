import numpy as np
import pandas as pd

MAX_TRIP_DURATION = 2_000

MIN_LATITUDE = 40.680
MIN_LONGITUDE = -74.02
CLIP_X = 0.09


def load_citibike_data(normalize: bool = False, private_features:bool=False, private_targets:bool=False):
    """
        Load and prepare CitiBike dataset, optionally normalize and clip features and targets to above pre-defined values.
        Args:
            normalize (bool): whether to normalize the data
            clip_features (bool): whether to clip feature values to pre-defined ranges (for differential privacy)
            clip_targets (bool): whether to clip target values to pre-defined ranges (for differential privacy)
        Returns:
            X, y: feature and target arrays
    """
    data = pd.read_csv("data/citibike_tripdata_201306.csv", index_col="Unnamed: 0")
    X = np.array(data.drop("tripduration", axis=1))
    y = np.array(data["tripduration"])
    assert np.sum(np.isnan(X)) + np.sum(np.isnan(y)) == 0
    # clip then privatise targets
    if private_targets:
        y_clipped = np.clip(y, a_min=0, a_max=2000)
        y_diff = y_clipped-y
        print(f"Clip coverage: targets={(1-np.count_nonzero(y_diff)/y.size)*100}%")
    # clip then privatise featuers
    if private_features:
        lats_start = np.clip(X[:, 0], a_min=MIN_LATITUDE, a_max=MIN_LATITUDE+CLIP_X)
        longs_start = np.clip(X[:, 1], a_min=MIN_LONGITUDE, a_max=MIN_LONGITUDE+CLIP_X)
        lats_end = np.clip(X[:, 2], a_min=MIN_LATITUDE, a_max=MIN_LATITUDE+CLIP_X)
        longs_end = np.clip(X[:, 3], a_min=MIN_LONGITUDE, a_max=MIN_LONGITUDE+CLIP_X)
        X_clipped = np.stack([lats_start, longs_start, lats_end, longs_end], axis=-1)
        x_diff = X_clipped-X
        print(f"Clip coverage: features={(1-np.count_nonzero(x_diff)/X.size)*100}")
    if normalize:
        pass
    return X, y

x, y = load_citibike_data(private_features=True, private_targets=True)
print(x.shape ,y.shape)



