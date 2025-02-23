import jax.numpy as jnp
from scipy.constants import physical_constants

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


def compute_relativistic_factors(
    energy: jnp.Array,
) -> tuple[jnp.Array, jnp.Array, jnp.Array]:
    """
    Computes the relativistic factors gamma, inverse gamma squared and beta for
    electrons.

    :param energy: Energy in eV.
    :return: gamma, igamma2, beta.
    """
    gamma = energy / electron_mass_eV
    igamma2 = jnp.where(gamma == 0.0, 0.0, 1 / gamma**2)
    beta = jnp.sqrt(1 - igamma2)

    return gamma, igamma2, beta
