import numpy as np
import time, sys
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import tensorflow_privacy
from typing import Tuple
from absl import flags, app
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
    compute_noise,
)

from dp_gp.approximate_inference.common_train_logic import (
    make_SVGP_model,
    simple_training_loop,
)
from dp_gp.approximate_inference.dp_gd_optimizer import VectorizedDPKerasAdamOptimizer
from utils import make_deterministic, plot_model

# gpu setup
gpus = tf.config.list_physical_devices('GPU')
print("\n GPU", gpus)
try:
  tf.config.experimental.set_memory_growth(gpus[0], True)
except:
  print("Couldn't set flexible memory growth, found GPU", gpus)
make_deterministic()
plt.rcParams["font.size"] = 22
dtype = np.float64
gpflow.config.set_default_float(dtype)

FLAGS = flags.FlagValues()
flags.DEFINE_integer("batch_size", 128, "batch size for training", flag_values=FLAGS)
flags.DEFINE_integer(
    "num_inducing", 100, "Number of inducing variables", flag_values=FLAGS
)
flags.DEFINE_float("lr", 1e-2, "learning rate", flag_values=FLAGS)
flags.DEFINE_integer("epochs", 70, "number of epochs", flag_values=FLAGS)
flags.DEFINE_float("epsilon", 2.0, "privacy budget", flag_values=FLAGS)
flags.DEFINE_float(
    "delta",
    1 / 4_000,
    "failure probability of privacy guarantee (for Gaussian mechanism)",
    flag_values=FLAGS,
)
flags.DEFINE_bool(
    "log_metrics",
    False,
    "whether to log metrics such as, test NLL/MSE, per-sample gradient norm and found hyperparameters",
    flag_values=FLAGS,
)
flags.DEFINE_float(
    "l2_clip", 10.0, "Clipping threshold as measured in the L2-norm", flag_values=FLAGS
)
flags.DEFINE_bool("plot_pred", True, "whether to plot resulting predictive distribution", flag_values=FLAGS)
flags.DEFINE_bool("apply_dp", True, "whether apply DP-SVI", flag_values=FLAGS)


def train_svi(
    batch_size: int,
    num_inducing: int,
    lr: float,
    epochs: int,
    epsilon: int,
    delta: int,
    l2_clip: float,
    apply_dp: float,
    plot_pred:bool =False,
):
    
    N = 5_000  # Number of training observations
    def func(x):
        return np.sin(x * 6) + 0.3 * np.cos(x * 2) + 0.5 * np.sin(15*x)

    X = np.random.rand(N, 1) * 2 - 1  # X values
    Y = func(X) + 0.2 * np.random.randn(N, 1)  # Noisy Y values
    X = X.astype(dtype)
    Y = Y.astype(dtype)

    n_test = 750
    idx=1_900
    sorted_ids = np.argsort(X, axis=0).flatten()
    test_idx = sorted_ids[idx-(n_test//2): idx+(n_test//2)]
    train_idx = main_list = list(set(sorted_ids) - set(test_idx))

    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]
    mse = tf.keras.losses.MeanSquaredError()

    # init inducing variables and model
    Z = np.random.uniform(-1, 1, size=(num_inducing, 1))
    m = make_SVGP_model(
        num_inducing=num_inducing, num_data=len(X), Z_init=Z, num_features=1
    )
    gpflow.utilities.print_summary(m)

    if apply_dp:
        NOISE_MULT = compute_noise(
            n=len(X_train),
            batch_size=batch_size,
            target_epsilon=epsilon,
            epochs=epochs,
            delta=delta,
            noise_lbd=0.1,
        )
        print(f"\n found noise_multiplier: {NOISE_MULT}")
        print("using private optimizer")
        opt = VectorizedDPKerasAdamOptimizer(
            l2_norm_clip=l2_clip,
            noise_multiplier=NOISE_MULT,
            num_microbatches=batch_size,
            learning_rate=lr,
        )
    else:
        print("using non private optimizer")
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    train_losses, train_elbos, _ = simple_training_loop(
        model=m,
        data=(X_train, y_train),
        optimizer=opt,
        batch_size=batch_size,
        epochs=epochs,
        logging_batch_freq=25,
        apply_dp=apply_dp,
    )
    if apply_dp:
        eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                n=len(X_train),
                batch_size=batch_size,
                noise_multiplier=NOISE_MULT,
                epochs=epochs,
                delta=delta,
        )
    if plot_pred:
        plot_model(m, (X_train, y_train), (X_test, y_test), is_svgp=True)
        plt.savefig("figures/predictive_dist.pdf")
    eps = np.inf if not apply_dp else eps
    print("\n Achieved privacy budget:", eps)
    nll_train = -1 * m.predict_log_density((X_train, y_train)).numpy().mean()
    nll_test = -1 * m.predict_log_density((X_test, y_test)).numpy().mean()
    y_pred_train, _ = m.predict_f(X_train)
    y_pred, _ = m.predict_f(X_test)
    rmse_train = np.sqrt(mse(y_train, y_pred_train).numpy())
    rmse_test = np.sqrt(mse(y_test, y_pred).numpy())
    print(f"train NLL: {nll_train:.4f}, RMSE: {rmse_train:.3f}")
    print(f"test NLL: {nll_test:.4f}, RMSE: {rmse_test:.3f}")

    l, sigma = m.kernel.lengthscales.numpy(), m.likelihood.variance.numpy()
    return (train_elbos, train_losses), (nll_test, rmse_test), (nll_train, rmse_train), l, sigma


def main(argv):
        FLAGS(sys.argv)
        train_svi(
        batch_size=FLAGS.batch_size,
        num_inducing=FLAGS.num_inducing,
        lr=FLAGS.lr,
        epochs=FLAGS.epochs,
        epsilon=FLAGS.epsilon,
        delta=FLAGS.delta,
        l2_clip=FLAGS.l2_clip,
        apply_dp=FLAGS.apply_dp,
        plot_pred=FLAGS.plot_pred
        )

if __name__ == "__main__":
    app.run(main)
