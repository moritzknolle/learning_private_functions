import os, sys, inspect, random, gc
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
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
L2_CLIP = 15.0
LR = 1e-2
EPS=2.0
DELTA = round(1 / 4_000, 25)
NUM_REPEATS = 5

eps_range = np.geomspace(1e-2, 15, num=15)
l2_clip_range = np.linspace(5.0, 75, 15)
seeds = [random.randint(0, 1000) for _ in range(NUM_REPEATS)]


elbos_scaled, (nll_test, rmse_test), (nll_train, rmse_train), l, sigma = train_svi(
    batch_size=BATCH_SIZE,
    num_inducing=50,
    lr=LR,
    epochs=EPOCHS,
    epsilon=0.1,
    delta=DELTA,
    l2_clip=L2_CLIP,
    apply_dp=False,
)

elbos_scaled_private, (nll_test, rmse_test), (nll_train, rmse_train), l, sigma = train_svi(
    batch_size=BATCH_SIZE,
    num_inducing=50,
    lr=LR,
    epochs=EPOCHS,
    epsilon=EPS,
    delta=DELTA,
    l2_clip=L2_CLIP,
    apply_dp=True,
)

plt.figure(figsize=(10, 6))
x_it = [25 * i for i in range(1, len(elbos_scaled)+1)]
plt.plot(x_it, elbos_scaled, lw=2.0, label='non-private')
plt.plot(x_it, elbos_scaled_private, lw=2.0, label='private')
plt.ylabel("ELBO")
plt.xlabel("iteration")
sns.despine()
plt.legend()
plt.ylim((-10_000, 100))
plt.tight_layout()
plt.savefig("/home/moritz/repositories/thesis/figures/dp_vfe/elbo_plot.pdf")