import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Tuple

dtype = np.float64


def make_deterministic(seed: int = 1234):
    """Makes PyTorch deterministic for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()


def plot_model(
    model,
    train_data,
    test_data=None,
    is_svgp: bool = True,
    title="",
    ms: str = "10",
    predict_y: bool = True,
    fig_size: Tuple = (10, 6),
):
    x, y = train_data
    plt.figure(figsize=fig_size)
    ylim = (-2.2, 2.2)
    plt.title(title) if title != "" else 0
    pX = np.linspace(-1.1, 1.1, 100, dtype=dtype)[:, None]  # Test locations
    pY, pYv = (
        model.predict_y(pX) if predict_y else model.predict_f(pX)
    )  # Predict Y values at test locations
    plt.plot(
        x, y, ".", label="training points", alpha=0.2, markersize=ms, color="#264185"
    )
    if test_data is not None:
        x_test, y_test = test_data
        plt.plot(
            x_test,
            y_test,
            ".",
            label="test points",
            alpha=0.2,
            markersize=ms,
            color="orange",
        )
    (line,) = plt.plot(
        pX, pY, lw=2.5, label="posterior mean", color="#A2D2FF", alpha=1.0
    )
    plt.fill_between(
        pX[:, 0],
        (pY - 2 * pYv**0.5)[:, 0],
        (pY + 2 * pYv**0.5)[:, 0],
        color="#A2D2FF",
        alpha=0.4,
        lw=1.5,
    )
    if is_svgp:
        Z = model.inducing_variable.Z.numpy()
        plt.plot(
            Z,
            np.repeat(ylim[0], Z.shape[0]),
            "k^",
            markersize="30",
            label="inducing locations",
            alpha=0.5,
            color="black",
        )
    plt.legend(loc="lower right")
    plt.ylim(ylim)
    sns.despine()
    plt.tight_layout()
