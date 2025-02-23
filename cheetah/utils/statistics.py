import jax.numpy as jnp


def unbiased_weighted_covariance(
    input1: jnp.Array, input2: jnp.Array, weights: jnp.Array, dim: int = None
) -> jnp.Array:
    """
    Compute the unbiased weighted covariance of two tensors.

    :param input1: Input tensor 1. (..., sample_size)
    :param input2: Input tensor 2. (..., sample_size)
    :param weights: Weights tensor. (..., sample_size)
    :param dim: Dimension along which to compute the covariance.
    :return: Unbiased weighted covariance. (..., 2, 2)
    """
    weighted_mean1 = jnp.sum(input1 * weights, dim=dim) / jnp.sum(weights, dim=dim)
    weighted_mean2 = jnp.sum(input2 * weights, dim=dim) / jnp.sum(weights, dim=dim)
    correction_factor = jnp.sum(weights, dim=dim) - jnp.sum(
        weights**2, dim=dim
    ) / jnp.sum(weights, dim=dim)
    covariance = jnp.sum(
        weights
        * (input1 - weighted_mean1.unsqueeze(-1))
        * (input2 - weighted_mean2.unsqueeze(-1)),
        dim=dim,
    ) / (correction_factor)
    return covariance


def unbiased_weighted_variance(
    input: jnp.Array, weights: jnp.Array, dim: int = None
) -> jnp.Array:
    """
    Compute the unbiased weighted variance of a tensor.

    :param input: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the variance.
    :return: Unbiased weighted variance.
    """
    weighted_mean = jnp.sum(input * weights, dim=dim) / jnp.sum(weights, dim=dim)
    correction_factor = jnp.sum(weights, dim=dim) - jnp.sum(
        weights**2, dim=dim
    ) / jnp.sum(weights, dim=dim)
    variance = jnp.sum(
        weights * (input - weighted_mean.unsqueeze(-1)) ** 2, dim=dim
    ) / (correction_factor)
    return variance


def unbiased_weighted_std(
    input: jnp.Array, weights: jnp.Array, dim: int = None
) -> jnp.Array:
    """
    Compute the unbiased weighted standard deviation of a tensor.

    :param input: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the standard deviation.
    :return: Unbiased weighted standard deviation.
    """
    return jnp.sqrt(unbiased_weighted_variance(input, weights, dim=dim))
