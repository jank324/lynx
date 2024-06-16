import jax.numpy as jnp

import lynx

beam_in = lynx.ParticleBeam.from_parameters(num_particles=100)


# Only Marker
def test_tracking_marker_only():
    segment = lynx.Segment([lynx.Marker(name="start")])

    beam_out = segment.track(beam_in)

    assert jnp.allclose(beam_out.particles, beam_in.particles)


# Only length-less elements between non-skippable elements
def test_tracking_lengthless_elements():
    segment = lynx.Segment(
        [
            lynx.Cavity(length=jnp.array([0.1]), voltage=jnp.array([1e6]), name="C2"),
            lynx.Marker(name="start"),
            lynx.Cavity(length=jnp.array([0.1]), voltage=jnp.array([1e6]), name="C1"),
        ]
    )

    _ = segment.track(beam_in)
