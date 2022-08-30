import os, sys, inspect, gc
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_privacy
import pandas as pd
from sklearn.model_selection import train_test_split
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
from dp_gp.dp_tools.mechanisms import laplace_mechanism, gauss_mechanism
make_deterministic()
dtype=np.float64
N = 250  # Number of training observations
def func(x):
    return np.sin(x * 6) + 0.3 * np.cos(x * 2) + 0.5 * np.sin(15*x)

X = np.random.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * np.random.randn(N, 1)  # Noisy Y values
X = X.astype(dtype)
Y = Y.astype(dtype)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print([a.shape for a in [X_train, X_test, y_train, y_test]])

mse = tf.keras.losses.MeanSquaredError()

x_clip = (-1.0, 1.0)
y_clip = (-2.0, 2.0)
eps_vals = [10, 20, 50, 100]
results = {"mode":[], "epsilon":[], "delta":[], "RMSE":[], "NLL":[]}

# non-private baseline
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
gpr_model = gpflow.models.GPR((X_train, y_train), kernel=gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(gpr_model.training_loss, gpr_model.trainable_variables, options=dict(maxiter=200))
y_pred, _ = gpr_model.predict_y(X_test)
rmse = np.sqrt(mse(y_test, y_pred).numpy())
nll = -1 * tf.reduce_mean(gpr_model.predict_log_density((X_test, y_test))).numpy()
results['mode'].append('non-private')
results['epsilon'].append(np.inf)
results['delta'].append(0.0)
results['RMSE'].append(rmse)
results['NLL'].append(nll)
del gpr_model
gc.collect()

# private models
for eps in eps_vals:
    for mode in ["Lap", "Gauss"]:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        if mode == 'Lap':
            delta = 0.0
            # for scalar-valued queries L1 and L2 norm are identical
            X_train = laplace_mechanism(np.clip(X_train, *x_clip), eps=eps, delta=delta, sens=2.0)
            y_train = laplace_mechanism(np.clip(y_train, *y_clip), eps=eps, delta=delta, sens=4.0)
        elif mode == 'Gauss':
            delta=0.01
            X_train = gauss_mechanism(np.clip(X_train, *x_clip), eps=eps, delta=delta, sens=2.0)
            y_train = gauss_mechanism(np.clip(y_train, *y_clip), eps=eps, delta=delta, sens=4.0)

        gpr_model = gpflow.models.GPR((X_train, y_train), kernel=gpflow.kernels.SquaredExponential())
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(gpr_model.training_loss, gpr_model.trainable_variables, options=dict(maxiter=200))
        y_pred, _ = gpr_model.predict_y(X_test)
        rmse = np.sqrt(mse(y_test, y_pred).numpy())
        nll = -1 * tf.reduce_mean(gpr_model.predict_log_density((X_test, y_test))).numpy()
        del gpr_model
        gc.collect()
        results['mode'].append(mode)
        results['epsilon'].append(eps)
        results['delta'].append(delta)
        results['RMSE'].append(rmse)
        results['NLL'].append(nll)


df = pd.DataFrame.from_dict(results)
print(df)
df.to_csv("/home/moritz/repositories/thesis/experiments/results/local_dp_toy_benchmark.csv")