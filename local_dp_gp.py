import gpflow
import numpy as np
import tensorflow as tf
from datasets import load_citibike_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_float("epsilon", 1.0, "privacy budget")
flags.DEFINE_boolean("private_features", True, "whether to privatise features")
flags.DEFINE_boolean("private_targets", True, "whether to privatise targets")
flags.DEFINE_integer("seed", 42, "random seed")
flags.DEFINE_enum("dataset", "citibike", ["citibike", "kung_sa"], "dataset") # TODO add support for Kung Sa Women dataset

delta= 1/5_000


def main(unused_arg_v):
    # Laplace mechanism
    if FLAGS.dataset == "citibike":
        X_lap, y_lap = load_citibike_data(eps=FLAGS.epsilon, delta=0.0, private_targets=FLAGS.private_targets, private_features=FLAGS.private_features, mechanism="lap")
        y_lap = y_lap[..., None]
    X_train, X_test, y_train, y_test = train_test_split(X_lap, y_lap, test_size=0.25, random_state=FLAGS.seed)

    length_scales = np.tile(0.3, X_train.shape[-1])
    print(length_scales)
    gpr_model = gpflow.models.GPR((X_train, y_train), kernel=gpflow.kernels.SquaredExponential(lengthscales=length_scales))
    gpr_model.likelihood.variance.assign(0.01)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(gpr_model.training_loss, gpr_model.trainable_variables, options=dict(maxiter=100))
    print(opt_logs)
    mu_pred, var_pred = gpr_model.predict_y(X_test)
    print(mu_pred)
    plt.hist(mu_pred, bins=15)
    plt.savefig("hist_preds.png")
    # Gauss mechanism
    # X_gauss, y_gauss = load_citibike_data(eps=eps, delta=delta, private_targets=True, mechanism="gauss")
    # X_train, X_test, y_train, y_test = train_test_split(X_gauss, y_gauss, test_size=0.25, random_state=FLAGS.seed)
    # gpr_model = gpflow.models.GPR((X_train, y_train), kernel=gpflow.kernels.SquaredExponential())
    # opt = gpflow.optimizers.Scipy()
    # opt_logs = opt.minimize(gp_model.training_loss, gp_model.trainable_variables, options=dict(maxiter=100))

if __name__ == "__main__":
    app.run(main)


