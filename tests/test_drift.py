import jax.numpy as jnp
import pytest

import lynx


def test_diverging_parameter_beam():
    """
    Test that that a parameter beam with sigma_px > 0 and sigma_py > 0 increases in
    size in both dimensions when travelling through a drift section.
    """
    drift = lynx.Drift(length=jnp.asarray(1.0))
    incoming_beam = lynx.ParameterBeam.from_parameters(
        sigma_px=jnp.asarray(2e-7), sigma_py=jnp.asarray(2e-7)
    )
    outgoing_beam = drift.track(incoming_beam)

    assert outgoing_beam.sigma_x > incoming_beam.sigma_x
    assert outgoing_beam.sigma_y > incoming_beam.sigma_y
    assert jnp.isclose(outgoing_beam.total_charge, incoming_beam.total_charge)


def test_diverging_particle_beam():
    """
    Test that that a particle beam with sigma_px > 0 and sigma_py > 0 increases in
    size in both dimensions when travelling through a drift section.
    """
    drift = lynx.Drift(length=jnp.asarray(1.0))
    incoming_beam = lynx.ParticleBeam.from_parameters(
        num_particles=jnp.asarray(1_000),
        sigma_px=jnp.asarray(2e-7),
        sigma_py=jnp.asarray(2e-7),
    )
    outgoing_beam = drift.track(incoming_beam)

    assert outgoing_beam.sigma_x > incoming_beam.sigma_x
    assert outgoing_beam.sigma_y > incoming_beam.sigma_y
    assert jnp.allclose(outgoing_beam.particle_charges, incoming_beam.particle_charges)


@pytest.mark.skip(
    reason="Requires rewriting Element and Beam member variables to be buffers."
)
def test_device_like_torch_module():
    """
    Test that when changing the device, Drift reacts like a `jnp.nn.Module`.
    """
    # There is no point in running this test, if there aren't two different devices to
    # move between
    if not jnp.cuda.is_available():
        return

    element = lynx.Drift(length=jnp.asarray(0.2), device="cuda")

    assert element.length.device.type == "cuda"

    element = element.cpu()

    assert element.length.device.type == "cpu"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_drift_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a drift with the `"bmadx"` tracking method
    match the results from Bmad-X.
    """
    incoming_beam = jnp.load(
        "tests/resources/bmadx/incoming.pt", weights_only=False
    ).to(dtype)
    drift = lynx.Drift(length=jnp.asarray(1.0), tracking_method="bmadx", dtype=dtype)

    # Run tracking
    outgoing_beam = drift.track(incoming_beam)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = jnp.load(
        "tests/resources/bmadx/outgoing_drift.pt", weights_only=False
    )

    assert jnp.allclose(
        outgoing_beam.particles,
        outgoing_bmadx.to(dtype),
        atol=1e-14 if dtype == jnp.float64 else 0.00001,
        rtol=1e-14 if dtype == jnp.float64 else 1e-6,
    )


def test_length_as_parameter():
    """Test that the drift length can be set as a `jnp.nn.Parameter`."""
    length = jnp.asarray(1.0)
    parameter = jnp.nn.Parameter(length)

    # Create to equal drifts, one with Tensor, one with Parameter
    drift = lynx.Drift(length=length)
    drift_parameter = lynx.Drift(length=parameter)

    incoming = lynx.ParameterBeam.from_parameters()
    outgoing = drift.track(incoming)
    outgoing_parameter = drift_parameter.track(incoming)

    # Check that all properties of the two outgoing beams are same
    for buffer, buffer_parameter in zip(
        outgoing.buffers(), outgoing_parameter.buffers()
    ):
        assert jnp.allclose(buffer, buffer_parameter)
