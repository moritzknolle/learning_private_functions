import numpy as np
import tensorflow as tf
import gpflow
from tqdm import tqdm
from typing import Tuple
import warnings

from dp_gp.approximate_inference.psg_svgp import SVGP_psg
from dp_gp.dp_tools.dp_gd_optimizer import (
    VectorizedDPKerasAdagradOptimizer,
    VectorizedDPKerasAdamOptimizer,
    VectorizedDPKerasSGDOptimizer,
)

dtype = np.float64
gpflow.config.set_default_float(dtype)
f64 = lambda x: np.array(x).astype(np.float64)


def make_SVGP_model(
    num_inducing: int,
    num_data: int,
    Z_init: np.ndarray = None,
    kernel=gpflow.kernels.SquaredExponential(),
    likelihood=gpflow.likelihoods.Gaussian(),
    num_features: int = 1,
    learnable_inducing_variables: bool = False,
):
    """Creates a variational free enery (VFE) GP approximation model.

    Args:
        num_inducing (int): Number of inducing points
        num_data (int): Number of samples in dataset
        kernel (fn): Kernel function for GP model
        likelihood (fn): Likelihood function for GP model
        learnable_inducing_variables (bool): Whether to learning inducing input locations or leave them fixed
    """
    if Z_init is None:
        Z_init = np.random.uniform(-1, 1, size=(num_inducing, num_features))
    assert Z_init.shape[0] == num_inducing
    model = SVGP_psg(kernel, likelihood, Z_init, num_data=num_data)
    gpflow.set_trainable(
        model.inducing_variable, False
    ) if not learnable_inducing_variables else 0
    return model


def optimization_step(
    model: gpflow.models.SVGP,
    batch: Tuple[tf.Tensor, tf.Tensor],
    optimizer: tf.keras.optimizers.Optimizer,
    apply_dp: bool = True,
):
    """Performs a single stochastic variatonal inference optimization update step given a model, mini-batch and optimizer. If apply_dp is set to True,
    per-sample gradients are computing in vectorized fashion and privatised before being applied.

    Args:
        model (gpflow.models.SVGP): an SVGP gpflow model, which inducing variables and hyperparameters we would like to learn
        batch (Tuple[tf.Tensor, tf.Tensor]): a mini-batch of data
        optimizer (tf.keras.optimizers.Optimizer): An optimizer instance
        apply_dp (bool): Whether to apply the Gaussian mechanism of differential privacy to the update step
    """

    if not isinstance(model, SVGP_psg):
        raise ValueError(
            "Computing per sample gradients requires model to be an instance of the SVGP_psg class"
        )
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
        batch_loss = tf.math.reduce_mean(loss)
        if apply_dp:
            dp_grads, norms = optimizer.get_gradients(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(dp_grads, model.trainable_variables))
        else:
            grads = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            norms = None
    del tape
    return batch_loss, norms


def simple_training_loop(
    model: gpflow.models.SVGP,
    data: Tuple[np.ndarray, np.ndarray],
    optimizer: tf.keras.optimizers.Optimizer,
    batch_size: int = 256,
    epochs: int = 5,
    logging_batch_freq: int = 10,
    apply_dp: bool = True,
    track_psg_norms: bool = False,
):
    """Runs an SGD-based VFE optimization precedure for GP model with given specified parameters.

    Args:
        model (gpflow.models.SVGP): model to train
        train_dataset (tf.data.Dataset): dataset to train model with
        optimizer (tf.keras.optimizers.Optimizer): optimizer to use for training
        batch_size (int): Batch size for training, larger batches means more accurate and computationally expensive gradient computations.
                            Larger batch sizes worsens privacy guarantee (worse subsampling amplification)
        epochs (int): Number of epochs to train for
        logging_batch_freq (int): Logging frequency (for progress bar)
        apply_dp (bool): whether to apply differential privacy
        track_psg_norms (bool): whether to track and return per-sample gradient norms
    """
    x_train, y_train = data
    if not isinstance(
        optimizer,
        (
            VectorizedDPKerasAdagradOptimizer,
            VectorizedDPKerasAdamOptimizer,
            VectorizedDPKerasSGDOptimizer,
        ),
    ):
        warnings.warn("Caution! Using non-differentially private optimizer")
        if apply_dp:
            raise ValueError(
                f"Can't apply differential privacy with non-private optimizer: {optimizer}"
            )
    losses, elbos = [], []
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (f64(x_train), f64(y_train))
    ).shuffle(x_train.shape[0])
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
    tf_optimization_step = tf.function(optimization_step)
    norm_vals = []
    with tqdm(total=epochs * len(train_dataset)) as pbar:
        for e in range(epochs):
            for i, (x_batch, y_batch) in enumerate(train_dataset):
                loss, norms = tf_optimization_step(
                    model, (x_batch, y_batch), optimizer, apply_dp=apply_dp
                )
                norm_vals.append(norms) if track_psg_norms else 0
                losses.append(loss)
                pbar.update()
                if i % logging_batch_freq == 0:
                    elbo = tf.math.reduce_mean(model.elbo(data))
                    elbos.append(elbo.numpy())
                    pbar.set_description(f"loss: {loss:.3f} ELBO: {elbo.numpy():.3f}")
    return losses, elbos, norm_vals
