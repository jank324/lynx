import jax.numpy as jnp

import lynx


def test_aperture_shape():
    """Test that the two aperture shapes produce differently shaped beams."""

    incoming = lynx.ParticleBeam.make_linspaced(num_particles=5)
    incoming.x = jnp.asarray([0.0, 2e-4, 2e-4, -2e-4, -2e-4])
    incoming.y = jnp.asarray([0.0, 3e-4, -3e-4, -3e-4, 3e-4])

    # Choose aperture size slightly larger than the beam width
    aperture = lynx.Aperture(
        x_max=jnp.asarray(2.2e-4), y_max=jnp.asarray(3.2e-4), shape="rectangular"
    )
    outgoing_rectangular = aperture.track(incoming)

    aperture.shape = "elliptical"
    outgoing_elliptical = aperture.track(incoming)

    assert jnp.allclose(outgoing_rectangular.survival_probabilities, jnp.ones(5))
    assert jnp.allclose(
        outgoing_elliptical.survival_probabilities,
        jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
