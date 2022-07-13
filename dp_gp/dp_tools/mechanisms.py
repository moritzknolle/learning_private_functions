import numpy as np
import matplotlib.pyplot as plt


def laplace_mechanism(inp_arr: np.ndarray, eps: float, delta:float, sens: float):
    """Applies the LaPlace mechanism to achieve (eps, 0) differential privacy for inp_arr
    Args:
        inp_arr (np.ndarray): input array to be privatised
        eps: privacy budget epsilon
        sens: global sensitivity, i.e. the maximal bound by which the contents of inp_arr can change
    """
    if delta != 0.0:
        raise ValueError("Laplace mechanism doesn't support non-zero values for delta")
    print(f"Adding independent Laplace noise with scale={sens/eps:.4f}")
    noise_sample = np.random.laplace(loc=0, scale=sens / eps, size=inp_arr.shape)
    priv_out = inp_arr + noise_sample
    return priv_out


def gauss_mechanism(inp_arr: np.ndarray, eps: float, delta: float, sens: float):
    """Applies the Gaussian mechanism to achieve (eps, delta) differential privacy for inp_arr
    Args:
        inp_arr (np.ndarray): input array to be privatised
        eps: privacy budget epsilon
        delta: failure probability by which the guarantee doesn't hold exactly
        sens: global sensitivity, i.e. the maximal bound by which the contents of inp_arr can change
    """
    print(f"Adding independent Gaussian noise with std={sens * (np.sqrt(2 * np.log(1.25 / delta))) / eps:4f}")
    noise_sample = np.random.normal(
        loc=0,
        scale=sens * (np.sqrt(2 * np.log(1.25 / delta))) / eps,
        size=inp_arr.shape,
    )
    priv_out = inp_arr + noise_sample
    return priv_out


def analytic_gauss_mechanism(
    inp_arr: np.ndarray, eps: float, delta: float, sens: float
):
    """Applies the analytic Gaussian mechanism (Balle and Wang, 2018) to achieve (eps, delta) differential privacy for inp_arr
    Args:
        inp_arr (np.ndarray): input array to be privatised
        eps: privacy budget epsilon
        delta: failure probability by which the guarantee doesn't hold exactly
        sens: global sensitivity, i.e. the maximal bound by which the contents of inp_arr can change
    """
    raise NotImplementedError()
