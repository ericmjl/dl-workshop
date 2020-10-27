from jax.nn.functions import softmax
from jax.scipy.special import expit
import jax.numpy as np
from jax.scipy import stats


def loglike_one_component(component_weight, component_mu, log_component_scale, datum):
    """Log likelihood of datum under one component of the mixture.

    Defined as the log likelihood of observing that datum from the component
    (i.e. log of component probability)
    added to the log likelihood of observing that datum
    under the Gaussian that belongs to that component.

    :param component_weight: Component weight, a scalar value between 0 and 1.
    :param component_mu: A scalar value.
    :param log_component_scale: A scalar value.
        Gets exponentiated before being passed into norm.logpdf.
    :returns: A scalar.
    """
    component_scale = np.exp(log_component_scale)
    return np.log(component_weight) + stats.norm.logpdf(
        datum, loc=component_mu, scale=component_scale
    )
