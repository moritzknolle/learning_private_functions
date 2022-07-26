import os, sys, inspect, random, time
from uci_datasets import all_datasets, Dataset
import tensorflow as tf
import numpy as np
import pandas as pd

# add parent directory to sys.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(currentdir))

from utils import make_deterministic
from dp_gp.approximate_inference.common_train_logic import (
    make_SVGP_model,
    simple_training_loop,
)

# for reproducible experiments
make_deterministic()
f64 = lambda x: np.array(x).astype(np.float64)

results = {"Dataset": [], "MSE (test)": [], "NLP (test)":[], "run-time (s)": []}
n_partitions = 3
datasets = [
    name
    for name, (n_observations, n_dimensions) in all_datasets.items()
    if n_observations < 1000
]
print("Performing benchmark on Datasets:", datasets)
mse = tf.keras.losses.MeanSquaredError()

for dataset_name in datasets:
    data = Dataset(dataset_name)
    start_time = time.time()
    mse_vals = []
    nlp_vals = []
    for n in range(n_partitions):
        x_train, y_train, x_test, y_test = data.get_split(split=n)
        N = x_train.shape[0]
        model = make_SVGP_model(
            num_inducing=100, num_data=N, num_features=x_train.shape[1]
        )
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        simple_training_loop(
            model=model, data=(x_train, y_train), optimizer=optimizer, epochs=50, batch_size=128, apply_dp=False
        )
        y_pred = model.predict_f(f64(x_test))
        test_mse = mse(y_test, y_pred).numpy()
        nlp = -1 * model.predict_log_density((x_test, y_test)).numpy().mean()
        mse_vals.append(test_mse)
        nlp_vals.append(nlp)
    results['Dataset'].append(dataset_name)
    results['MSE (test)'].append(f"{np.mean(mse_vals):.4f} ± {np.std(mse_vals):.4f}")
    results['NLP (test)'].append(f"{np.mean(nlp_vals):.4f} ± {np.std(nlp_vals):.4f}")
    results['run-time (s)'].append((time.time() - start_time)/n_partitions)


df = pd.DataFrame.from_dict(results)
df.to_csv("experiments/uci_results.csv")
print(df)
