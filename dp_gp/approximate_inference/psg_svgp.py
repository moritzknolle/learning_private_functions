import tensorflow as tf
import gpflow
from gpflow.base import RegressionData

class SVGP_psg(gpflow.models.svgp.SVGP):
    """ SVGP subclass that returns a vector of per-sample ELBO's need to calculate per-sample gradients for DP-SGD.
    """

    def elbo(self, data: RegressionData, reduce:bool=False) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        if reduce:
            out = tf.reduce_sum(var_exp) * scale - kl
        else:
            out = var_exp * scale - kl
        return out