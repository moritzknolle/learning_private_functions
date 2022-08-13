import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.neighbors import KernelDensity
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


def plot_grad_norms(
    norms: list, max_norm: int, cmap: str = "viridis", n_sample=100, vis_mult=100
):
    cmap = mpl.cm.get_cmap(cmap)
    fig = plt.figure(figsize=(5, 10))
    kde = KernelDensity(kernel="tophat", bandwidth=2.0)
    X_plot = np.linspace(0, max_norm, 500)[:, None]

    for i, norm_vals in enumerate(norms):
        if i % n_sample == 0:
            kde.fit(norm_vals[:, None])
            log_dens = kde.score_samples(X_plot)
            color = cmap(i / len(norms))
            plt.plot(X_plot[:, 0], vis_mult * np.exp(log_dens) + i, color=color)
            plt.fill_between(
                X_plot[:, 0],
                vis_mult * np.exp(log_dens) + i,
                i,
                color=color,
                zorder=300 - i,
            )

    plt.xlim((0, max_norm))
    plt.ylim((-1, len(norms) + 50))
