import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

dtype = np.float64

def make_deterministic(seed: int = 1234):
    """Makes PyTorch deterministic for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

def plot_model(model, train_data, is_svgp:bool=True, title=""):
    x, y = train_data
    plt.figure(figsize=(12, 6))
    ylim=(-2.2, 2.2)
    plt.title(title) if title != "" else 0
    pX = np.linspace(-1.1, 1.1, 100, dtype=dtype)[:, None]  # Test locations
    pY, pYv = model.predict_y(pX)  # Predict Y values at test locations
    plt.plot(x, y, ".", label="training points", alpha=0.2, markersize='10',color='#264185')
    (line,) = plt.plot(pX, pY, lw=2.5, label="posterior mean", color='#A2D2FF', alpha=1.0)
    plt.fill_between(
        pX[:, 0],
        (pY - 2 * pYv ** 0.5)[:, 0],
        (pY + 2 * pYv ** 0.5)[:, 0],
        color='#A2D2FF',
        alpha=0.4,
        lw=1.5,
    )
    if is_svgp:
        Z = model.inducing_variable.Z.numpy()
        plt.plot(Z, np.repeat(ylim[0], Z.shape[0]), 'k^', markersize='30', label="inducing locations", alpha=0.5, color='orange')
    plt.legend(loc="lower right")
    plt.ylim(ylim)
    sns.despine()
    plt.tight_layout()