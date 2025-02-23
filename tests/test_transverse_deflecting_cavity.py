import jax.numpy as jnp
import pytest

import lynx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_transverse_deflecting_cavity_bmadx_tracking(dtype):
    """
    Test that the results of tracking through a TDC with the `"bmadx"` tracking method
    match the results from Bmad-X.
    """
    incoming_beam = jnp.load(
        "tests/resources/bmadx/incoming.pt", weights_only=False
    ).to(dtype)
    tdc = lynx.TransverseDeflectingCavity(
        length=jnp.asarray(1.0),
        voltage=jnp.asarray(1e7),
        phase=jnp.asarray(0.2, dtype=dtype),
        frequency=jnp.asarray(1e9),
        tracking_method="bmadx",
        dtype=dtype,
    )

    # Run tracking
    outgoing_beam = tdc.track(incoming_beam)

    # Load reference result computed with Bmad-X
    outgoing_bmadx = jnp.load(
        "tests/resources/bmadx/outgoing_transverse_deflecting_cavity.pt",
        weights_only=False,
    )

    assert jnp.allclose(
        outgoing_beam.particles,
        outgoing_bmadx.to(dtype),
        atol=1e-14 if dtype == jnp.float64 else 0.00001,
        rtol=1e-14 if dtype == jnp.float64 else 1e-6,
    )


def test_transverse_deflecting_cavity_energy_length_vectorization():
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when the input beam's energy and the TDC's length are vectorised.
    """
    incoming_beam = lynx.ParticleBeam.from_parameters(
        num_particles=jnp.asarray(10_000),
        sigma_px=jnp.asarray(2e-7),
        sigma_py=jnp.asarray(2e-7),
        energy=jnp.asarray([50e6, 60e6]),
    )
    tdc = lynx.TransverseDeflectingCavity(
        length=jnp.asarray(1.0),
        voltage=jnp.asarray([[1e7], [2e7], [3e7]]),
        phase=jnp.asarray(0.4),
        frequency=jnp.asarray(1e9),
        tracking_method="bmadx",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == jnp.Size([3, 2])


def test_transverse_deflecting_cavity_energy_phase_vectorization():
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when the input beam's energy and the TDC's phase are vectorised.
    """
    incoming_beam = lynx.ParticleBeam.from_parameters(
        num_particles=jnp.asarray(10_000),
        sigma_px=jnp.asarray(2e-7),
        sigma_py=jnp.asarray(2e-7),
        energy=jnp.asarray([50e6, 60e6]),
    )
    tdc = lynx.TransverseDeflectingCavity(
        length=jnp.asarray(1.0),
        voltage=jnp.asarray(1e7),
        phase=jnp.asarray([[0.6], [0.5], [0.4]]),
        frequency=jnp.asarray(1e9),
        tracking_method="bmadx",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == jnp.Size([3, 2])


def test_transverse_deflecting_cavity_energy_frequency_vectorization():
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when the input beam's energy and the TDC's frequency are vectorised.
    """
    incoming_beam = lynx.ParticleBeam.from_parameters(
        num_particles=jnp.asarray(10_000),
        sigma_px=jnp.asarray(2e-7),
        sigma_py=jnp.asarray(2e-7),
        energy=jnp.asarray([50e6, 60e6]),
    )
    tdc3 = lynx.TransverseDeflectingCavity(
        length=jnp.asarray(1.0),
        voltage=jnp.asarray(1e7),
        phase=jnp.asarray(0.4),
        frequency=jnp.asarray([[1e9], [2e9], [3e9]]),
        tracking_method="bmadx",
    )

    _ = tdc3.track(incoming_beam)

    assert _.particles.shape[:-2] == jnp.Size([3, 2])


def test_transverse_deflecting_cavity_all_parameters_vectorization():
    """
    Test that vectorised tracking through a TDC throws now exception and outputs the
    correct shape, when all parameters are vectorised.
    """
    incoming_beam = lynx.ParticleBeam.from_parameters(
        num_particles=jnp.asarray(10_000),
        sigma_px=jnp.asarray(2e-7),
        sigma_py=jnp.asarray(2e-7),
        energy=jnp.asarray([50e6, 60e6]),
    )
    tdc = lynx.TransverseDeflectingCavity(
        length=jnp.asarray(1.0),
        voltage=jnp.ones([4, 1, 1, 1]) * 1e7,
        phase=jnp.ones([1, 3, 1, 1]) * 0.4,
        frequency=jnp.ones([1, 1, 2, 1]) * 1e9,
        tracking_method="bmadx",
    )

    outgoing_beam = tdc.track(incoming_beam)

    assert outgoing_beam.particles.shape[:-2] == jnp.Size([4, 3, 2, 2])


def test_tracking_inactive_in_segment():
    """
    Test that tracking through a `Segment` that contains an inactive
    `TransverseDeflectingCavity` does not throw an exception. This was an issue in #290.
    """
    segment = lynx.Segment(
        elements=[lynx.TransverseDeflectingCavity(length=jnp.asarray(1.0))]
    )
    beam = lynx.ParticleBeam.from_parameters()

    segment.track(beam)
