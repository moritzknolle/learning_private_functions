import os, sys, inspect, random, gc
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.models.svgp import SVGP as SVGPBase
from gpflow.optimizers.natgrad import NaturalGradient as NaturalGradientBase
from itertools import tee

# add parent directory to sys.path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(currentdir))
from utils import make_deterministic
from dp_gp.approximate_inference.psg_svgp import SVGP_psg
from dp_gp.approximate_inference.nat_grad_opt import NaturalGradient
make_deterministic()

#create synthetic dataset
N, D = 500, 2
batch_size = 50
# inducing points
M = 10
# synthetic training data
x = np.random.uniform(size=(N, D))
y = np.sin(10 * x[:, :1]) + 5 * x[:, 1:] ** 2
data = (x, y)
inducing_variable = tf.random.uniform((M, D))
adam_learning_rate = 0.01
# create dataset iterator
data_minibatch = (
tf.data.Dataset.from_tensor_slices(data)
.repeat()
.shuffle(N)
.batch(batch_size)
)
data_minibatch_it = iter(data_minibatch)

# instatiate models
svgp = SVGP_psg(
kernel=gpflow.kernels.Matern52(),
likelihood=gpflow.likelihoods.Gaussian(),
inducing_variable=inducing_variable,
num_data=x.shape[0]
)
svgp_base = SVGPBase(
    kernel=gpflow.kernels.Matern52(),
    likelihood=gpflow.likelihoods.Gaussian(),
    inducing_variable=inducing_variable,
    num_data=x.shape[0]
)
# our variatonal parameters we're trying to learn
variational_params = [(svgp.q_mu, svgp.q_sqrt)]
variatonal_params_base = [(svgp_base.q_mu, svgp_base.q_sqrt)]

def test_svgp_per_sample_elbo():
    assert svgp.elbo(data).numpy().flatten().shape[0] == N

def test_svgp_log_prior():
    assert np.allclose(svgp.log_prior_density(), svgp_base.log_prior_density(), rtol=1e-4)

def test_svgp_elbo_full_dataset():
    assert np.allclose(tf.reduce_mean(svgp.elbo(data)), svgp_base.elbo(data), rtol=1e-4)

def test_svgp_elbo_minibatch():
    data_batch = next(data_minibatch_it)
    assert np.allclose(tf.reduce_mean(svgp.elbo(data_batch)), svgp_base.elbo(data_batch), rtol=1e-4)

def test_svgp_train_loss():
    data_batch = next(data_minibatch_it)
    assert np.allclose(tf.reduce_mean(svgp.training_loss(data_batch)), svgp_base.training_loss(data_batch), rtol=1e-4)

def test_svgp_optim(n_iter=25):
    data_it, base_data_it = tee(data_minibatch_it)
    natgrad_opt = NaturalGradient(gamma=0.1)
    natgrad_opt_base = NaturalGradientBase(gamma=0.1)
    batch, batch_base = next(data_it), next(base_data_it)
    assert np.allclose(batch[0], batch_base[0], rtol=1e-4)
    assert np.allclose(batch[1], batch_base[1], rtol=1e-4)
    svgp_objective = svgp.training_loss_closure(data_it)
    svgp_base_objective = svgp_base.training_loss_closure(base_data_it)
    assert np.allclose(tf.reduce_mean(svgp_objective()), svgp_base_objective(), rtol=1e-4)
    for i in range(n_iter):
        natgrad_opt.minimize(svgp_objective, var_list=variational_params, reduce_loss=True)
        natgrad_opt_base.minimize(svgp_base_objective, var_list=variatonal_params_base)
    assert np.allclose(tf.reduce_mean(svgp.elbo(data)), svgp_base.elbo(data), rtol=1e-4)
