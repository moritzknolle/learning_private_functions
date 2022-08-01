import os, sys, inspect, random, time, gc
from uci_datasets import all_datasets, Dataset
import tensorflow as tf
import numpy as np
import gpflow
import pandas as pd
from absl import flags, app
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

# add parent directory to sys.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(currentdir))

from utils import make_deterministic
from dp_gp.dp_tools.mechanisms import gauss_mechanism, laplace_mechanism
from dp_gp.approximate_inference.common_train_logic import (
    make_SVGP_model,
    simple_training_loop,
)

# Hyperparameters
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size for training")
flags.DEFINE_integer("num_inducing", 512, "Number of inducing variables")
flags.DEFINE_float("lr", 0.01, "learning rate")
flags.DEFINE_integer("epochs", 50, "number of epochs")
flags.DEFINE_bool("save_results", True, "whether to dump results to a .csv file")
flags.DEFINE_integer("max_observations", 300, "max number of observations per dataset. Used to select suitable datasets from UCI repository")
flags.DEFINE_integer("n_folds", 3, "number of folds to perform for cross-validation")
flags.DEFINE_float("epsilon", 1.0, "privacy budget")
flags.DEFINE_float("delta", 0.01, "failure probability of privacy guarantee (for Gaussian mechanism)")

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
    # datasets = [
    #      name
    #      for name, (n_observations, n_dimensions) in all_datasets.items()
    #      if n_observations > FLAGS.max_observations
    # ]
    datasets = ['concrete']
    print("Performing benchmark on Datasets:", datasets)
    mse = tf.keras.losses.MeanSquaredError()

    for dataset_name in datasets:
        data = Dataset(dataset_name)
        for mode in ["SVGP", "Local-DP"]:
            start_time = time.time()
            nlp_vals = []
            mse_vals = []
            model = None
            gc.collect()
            for n in range(FLAGS.n_folds):
                x_train, y_train, x_test, y_test = data.get_split(split=n)
                if mode == "Local-DP":
                    x_sens = np.max(x_train) - np.min(x_train)
                    y_sens = np.max(y_train) - np.min(y_train)
                    print(f"\n \n ...... sens x: {x_sens}, y:{y_sens}")
                    x_train = laplace_mechanism(x_train, eps=FLAGS.epsilon, delta=0.0, sens=x_sens)
                    y_train = laplace_mechanism(y_train, eps=FLAGS.epsilon, delta=0.0, sens=y_sens)
                N = x_train.shape[0]
                feature_dim = x_train.shape[1]
                lengthscales=np.random.uniform(0.1, 5.0, size=(feature_dim))
                kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
                model = make_SVGP_model(
                    num_inducing=FLAGS.num_inducing, num_data=N, num_features=feature_dim, kernel=kernel
                )
                print_summary(model)
                optimizer = tf.keras.optimizers.Adam(lr=1e-3)
                simple_training_loop(
                    model=model,
                    data=(f64(x_train), f64(y_train)),
                    optimizer=optimizer,
                    epochs=FLAGS.epochs,
                    batch_size=FLAGS.batch_size,
                    apply_dp=False,
                )
                y_pred, _ = model.predict_f(f64(x_test))
                y_pred = np.array(y_pred)
                #print("y_test", y_test)
                #print("\n y_pred", y_pred)
                test_mse = mse(y_test, y_pred).numpy()
                nlp = -1 * model.predict_log_density((f64(x_test), f64(y_test))).numpy().mean()
                mse_vals.append(np.sqrt(test_mse))
                nlp_vals.append(nlp)
                plt.scatter(y_test, y_pred)
                plt.xlabel("Prices")
                plt.ylabel("Predicted prices")
                plt.title("Prices vs Predicted prices")
                plt.savefig(f"figures/price_preds_{n}.png")
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
