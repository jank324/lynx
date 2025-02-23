import jax.numpy as jnp

from lynx.utils import unbiased_weighted_covariance, unbiased_weighted_variance


def test_unbiased_weighted_variance_with_single_element():
    """Test that the variance is NaN when there is only one element."""
    data = jnp.asarray([42.0])
    weights = jnp.asarray([1.0])

    computed_variance = unbiased_weighted_variance(data, weights)

    assert jnp.isnan(computed_variance)


def test_unbiased_weighted_variance_with_same_weights():
    """
    Test that the weighted variance with all weights the same equals the unweighted
    variance implementated in PyTorch.
    """
    data = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0])

    expected_variance = jnp.var(data, unbiased=True)
    computed_variance = unbiased_weighted_variance(data, weights)

    assert jnp.allclose(computed_variance, expected_variance)


def test_unbiased_weighted_variance_with_different_weights():
    """Test that the variance is computed when some weights are different."""
    data = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = jnp.asarray([0.5, 0.5, 0.5, 0, 0])

    expected_variance = jnp.var(jnp.asarray([1.0, 2.0, 3.0]), unbiased=True)
    computed_variance = unbiased_weighted_variance(data, weights)

    assert jnp.allclose(computed_variance, expected_variance)


def test_unbiased_weighted_variance_with_zero_weights():
    """Test that the variance is NaN when all weights are zero."""
    data = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0])

    computed_variance = unbiased_weighted_variance(data, weights)

    assert jnp.isnan(computed_variance)


def test_unbiased_weighted_variance_with_small_numbers():
    """Test that the variance is correct for small numbers."""
    data = jnp.asarray([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], dtype=jnp.float32)
    weights = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    expected_variance = jnp.var(data, unbiased=True)
    computed_variance = unbiased_weighted_variance(data, weights)

    assert jnp.allclose(computed_variance, expected_variance)


def test_unbiased_weighted_covariance_reduced_to_variance():
    """
    Test that the covariance computation is correctly reduced to the variance when both
    inputs are the same.
    """
    data = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = jnp.asarray([0.5, 1.0, 1.0, 0.9, 0.9])

    variance = unbiased_weighted_variance(data, weights)
    covariance = unbiased_weighted_covariance(data, data, weights)

    assert jnp.allclose(covariance, variance)
