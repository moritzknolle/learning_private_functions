import os, sys, inspect, random, time, gc
from uci_datasets import all_datasets, Dataset
import tensorflow as tf
import numpy as np
import gpflow
import pandas as pd
from absl import flags, app
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
from sklearn.preprocessing import MinMaxScaler
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
    compute_noise,
)
# gpu setup
gpus = tf.config.list_physical_devices('GPU')
print("\n GPU", gpus)
try:
  tf.config.experimental.set_memory_growth(gpus[0], True)
except:
  print("Couldn't set flexible memory growth, found GPU", gpus)


# add parent directory to sys.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(currentdir))

from utils import make_deterministic
from dp_gp.dp_tools.mechanisms import gauss_mechanism, laplace_mechanism
from dp_gp.approximate_inference.common_train_logic import (
    make_SVGP_model,
    simple_training_loop,
)
from dp_gp.approximate_inference.dp_gd_optimizer import VectorizedDPKerasAdamOptimizer

# Hyperparameters
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size for training")
flags.DEFINE_integer("num_inducing", 512, "Number of inducing variables")
flags.DEFINE_float("lr", 1e-2, "learning rate")
flags.DEFINE_float("l2_clip", 10.0, "clipping treshold for DP-SGD")
flags.DEFINE_integer("epochs", 200, "number of epochs")
flags.DEFINE_bool("save_results", True, "whether to dump results to a .csv file")
flags.DEFINE_integer("n_folds", 5, "number of folds to perform for cross-validation")
flags.DEFINE_float("epsilon", 2.0, "privacy budget")
flags.DEFINE_float("delta_ldp", 0.01, "failure probability of privacy guarantee (for Gaussian mechanism)")

# for reproducible experiments
make_deterministic(seed=42)
f64 = lambda x: np.array(x).astype(np.float64)

def main(argv):
    results = {
        "Dataset": [],
        "Model":[],
        "epsilon": [],
        "RMSE (test)": [],
        "NLL (test)": [],
        "run-time (s)": [],
    }
    datasets = ['energy', 'bike', 'elevators', 'parkinsons']
    print("Performing benchmark on Datasets:", datasets)
    mse = tf.keras.losses.MeanSquaredError()

    for dataset_name in datasets:
        data = Dataset(dataset_name)
        for mode in ["SVGP", "Local-DP", "DP-SVGP"]:
            start_time = time.time()
            nlp_vals = []
            mse_vals = []
            model = None
            gc.collect()
            for n in range(FLAGS.n_folds):
                # get cross-val splot
                x_train, y_train, x_test, y_test = data.get_split(split=n)
                # normalize (in reality max and min values here would also have to be computed in a private fashion)
                x_scaler = MinMaxScaler().fit(x_train)
                y_scaler = MinMaxScaler().fit(y_train)
                x_train, x_test = x_scaler.transform(x_train), x_scaler.transform(x_test)
                y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)
                # setup training for different modes
                if mode == "Local-DP":
                    # if we were complete kosher these max and min calculations would have to be privatised as well
                    x_sens = np.linalg.norm(np.ones(shape=(x_train.shape[1])), ord=1)
                    y_sens = np.amax(y_train) - np.amin(y_train) # no need to calculate vector valued sensitivity for scalars
                    print(f"\n \n ...... sens x: {x_sens}, y:{y_sens}")
                    x_train = laplace_mechanism(x_train, eps=FLAGS.epsilon/2, delta=0.0, sens=x_sens)
                    y_train = laplace_mechanism(y_train, eps=FLAGS.epsilon/2, delta=0.0, sens=y_sens)
                if mode == "DP-SVGP":
                    NOISE_MULT = compute_noise(
                            n=len(x_train),
                            batch_size=FLAGS.batch_size,
                            target_epsilon=FLAGS.epsilon,
                            epochs=FLAGS.epochs,
                            delta= round(1/(len(x_train)), 5),
                            noise_lbd=0.05,
                        )
                    print(f"found noise multiplier: {NOISE_MULT} for target epsilon: {FLAGS.epsilon}")
                    optimizer = VectorizedDPKerasAdamOptimizer(l2_norm_clip=FLAGS.l2_clip, noise_multiplier=NOISE_MULT, lr=FLAGS.lr)
                elif mode == "Local-DP" or mode == "SVGP":
                    optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
                N = x_train.shape[0]
                feature_dim = x_train.shape[1]
                lengthscales=np.random.uniform(0.1, 5.0, size=(feature_dim))
                kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
                model = make_SVGP_model(
                    num_inducing=FLAGS.num_inducing, num_data=N, num_features=feature_dim, kernel=kernel, learnable_inducing_variables=True
                )
                print_summary(model)
                apply_dp = (mode == "DP-SVGP")
                print("applying dp:", apply_dp)
                # train model
                simple_training_loop(
                    model=model,
                    data=(f64(x_train), f64(y_train)),
                    optimizer=optimizer,
                    epochs=FLAGS.epochs,
                    batch_size=FLAGS.batch_size,
                    apply_dp=apply_dp,
                )
                # get val-set metrics
                y_pred, _ = model.predict_f(f64(x_test))
                y_pred = np.array(y_pred)
                test_mse = mse(y_test, y_pred).numpy()
                nlp = -1 * model.predict_log_density((f64(x_test), f64(y_test))).numpy().mean()
                mse_vals.append(np.sqrt(test_mse))
                nlp_vals.append(nlp)
                print("\n \n rmse:", np.sqrt(test_mse), "nll", nlp)
            # log val-set metrics over splits
            if mode == "SVGP":
                results['epsilon'].append(np.inf)
            else:
                results['epsilon'].append(FLAGS.epsilon)
            results["Dataset"].append(dataset_name)
            results["Model"].append(mode)
            results["RMSE (test)"].append(
                f"{np.mean(mse_vals):.3f} ± {np.std(mse_vals):.3f}"
            )
            results["NLL (test)"].append(
                f"{np.mean(nlp_vals):.3f} ± {np.std(nlp_vals):.3f}"
            )
            results["run-time (s)"].append((time.time() - start_time) / FLAGS.n_folds)

    df = pd.DataFrame.from_dict(results)
    df.to_csv("experiments/uci_results.csv") if FLAGS.save_results else 0
    print(df)

if __name__ == "__main__":
    app.run(main)
