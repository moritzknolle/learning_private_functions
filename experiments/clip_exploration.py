import os, sys, inspect, random, gc
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import tensorflow_privacy
import pandas as pd
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
from dp_svfe import train_svi

make_deterministic()
plt.rcParams["font.size"] = 22
dtype = np.float64
gpflow.config.set_default_float(dtype)

BATCH_SIZE = 128
EPOCHS = 70
LR = 1e-2
DELTA = round(1 / 4_000, 25)
NUM_REPEATS = 5

l2_clip_range = np.geomspace(5, 100, num=15)
eps_range = [0.1, 1.0, 2.0]
seeds = [random.randint(0, 1000) for _ in range(NUM_REPEATS)]

results = {"NLL(train)":[], "RMSE(train)":[], "NLL(test)":[], "RMSE(test)":[], "l_mean":[], "l_std":[], "sigma_mean":[], "sigma_std":[], "eps":[], "l2_clip":[]}

# results for varying privacy budgets
for l2_clip in l2_clip_range:
    for e, eps in enumerate(eps_range):
        for l2_clip in l2_clip_range:
            train_nll_vals, test_nll_vals, train_rmse_vals, test_rmse_vals, l_vals, sigma_vals = [], [], [], [], [], []
            for i in range(NUM_REPEATS):
                make_deterministic(seeds[i])
                elbos_scaled, (nll_test, rmse_test), (nll_train, rmse_train), (l, s, sigma) = train_svi(
                    batch_size=BATCH_SIZE,
                    num_inducing=50,
                    lr=LR,
                    epochs=EPOCHS,
                    epsilon=eps,
                    delta=DELTA,
                    l2_clip=l2_clip,
                    apply_dp=True,
                )
                train_nll_vals.append(nll_train)
                train_rmse_vals.append(rmse_train)
                test_nll_vals.append(nll_test)
                test_rmse_vals.append(rmse_test)
                l_vals.append(l)
                sigma_vals.append(sigma)
            results['NLL(train)'].append(np.mean(train_nll_vals))
            results['RMSE(train)'].append(np.mean(train_rmse_vals))
            results['NLL(test)'].append(np.mean(test_nll_vals))
            results['RMSE(test)'].append(np.mean(test_rmse_vals))
            results['l_mean'].append(np.mean(l_vals))
            results['l_std'].append(np.std(l_vals))
            results['sigma_mean'].append(np.mean(sigma_vals))
            results['sigma_std'].append(np.std(sigma_vals))
            results['eps'].append(eps)
            results['l2_clip'].append(l2_clip)

df = pd.DataFrame().from_dict(results)
print(df)
df.to_csv("clip_experiment_results.csv")
