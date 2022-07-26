import tensorflow as tf
import numpy as np
import random

def make_deterministic(seed: int = 1234):
    """Makes PyTorch deterministic for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()

