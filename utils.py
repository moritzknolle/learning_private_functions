import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Tuple

dtype = np.float64


def make_deterministic(seed: int = 1234):
    """Makes PyTorch deterministic for reproducibility.
        Args:
            seed (int): seed value to set
    """
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
    alpha=0.2,
    fig_size: Tuple = (10, 6),
    show_legend=True,
):
    """Produces a nice plot of a Gaussian process model.
        Args:
            model (gpflow.model): a GPflow model instance
            train_data (tuple[np.ndarray, np.ndarray]): tuple containing the training data to plot (in blue)
            test_data (tuple[np.ndarray, np.ndarray]): tuple containing the test data to plot (in orange)
            is_svgp (bool) : if model is an SVGP model, this will cause inducing inputs to be plotted
            title (str): matplotlib title for figure
            ms (str(int)): marker-size for data scatter plot
            predict_y (bool): whether to predict observations 'y' or function values 'f'
            alpha (float): alpha (opacity) value for data points
            fig_size (tuple(int, int)): figure size
            show_legend (bool): whether to plot a legend
    """ 
    x, y = train_data
    plt.figure(figsize=fig_size)
    ylim = (-2.2, 2.2)
    plt.title(title) if title != "" else 0
    pX = np.linspace(-1.1, 1.1, 100, dtype=dtype)[:, None]  # Test locations
    pY, pYv = (
        model.predict_y(pX) if predict_y else model.predict_f(pX)
    )  # Predict Y values at test locations
    plt.plot(
        x, y, ".", label="training points", alpha=alpha, markersize=ms, color="#264185"
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
    plt.legend(loc="lower right") if show_legend else 0
    plt.ylim(ylim)
    sns.despine()
    plt.tight_layout()
