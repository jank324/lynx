import jax.numpy as jnp
import pytest

import lynx


@pytest.mark.xfail  # TODO: Fix this
def test_drift_end():
    """
    Test that at the end of a split drift the result is the same as at the end of the
    original drift.
    """
    original_drift = lynx.Drift(length=jnp.array([2.0]))
    split_drift = lynx.Segment(original_drift.split(resolution=jnp.array(0.1)))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_drift.track(incoming_beam)
    outgoing_beam_split = split_drift.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


@pytest.mark.xfail  # TODO: Fix this
def test_quadrupole_end():
    """
    Test that at the end of a split quadrupole the result is the same as at the end of
    the original quadrupole.
    """
    original_quadrupole = lynx.Quadrupole(length=jnp.array([0.2]), k1=jnp.array([4.2]))
    split_quadrupole = lynx.Segment(original_quadrupole.split(resolution=0.01))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_quadrupole.track(incoming_beam)
    outgoing_beam_split = split_quadrupole.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_cavity_end():
    """
    Test that at the end of a split cavity the result is the same as at the end of
    the original cavity.
    """
    original_cavity = lynx.Cavity(
        length=jnp.array([1.0377]),
        voltage=jnp.array([0.01815975e9]),
        frequency=jnp.array([1.3e9]),
        phase=jnp.array([0.0]),
    )
    split_cavity = lynx.Segment(original_cavity.split(resolution=0.1))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_cavity.track(incoming_beam)
    outgoing_beam_split = split_cavity.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_solenoid_end():
    """
    Test that at the end of a split solenoid the result is the same as at the end of
    the original solenoid.
    """
    original_solenoid = lynx.Solenoid(length=jnp.array([0.2]), k=jnp.array([4.2]))
    split_solenoid = lynx.Segment(original_solenoid.split(resolution=0.01))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_solenoid.track(incoming_beam)
    outgoing_beam_split = split_solenoid.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_dipole_end():
    """
    Test that at the end of a split dipole the result is the same as at the end of
    the original dipole.
    """
    original_dipole = lynx.Dipole(length=jnp.array([0.2]), angle=jnp.array([4.2]))
    split_dipole = lynx.Segment(original_dipole.split(resolution=0.01))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_dipole.track(incoming_beam)
    outgoing_beam_split = split_dipole.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_undulator_end():
    """
    Test that at the end of a split undulator the result is the same as at the end of
    the original undulator.
    """
    original_undulator = lynx.Undulator(length=jnp.array([3.142]))
    split_undulator = lynx.Segment(original_undulator.split(resolution=0.1))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_undulator.track(incoming_beam)
    outgoing_beam_split = split_undulator.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


@pytest.mark.xfail  # TODO: Fix this
def test_horizontal_corrector_end():
    """
    Test that at the end of a split horizontal corrector the result is the same as at
    the end of the original horizontal corrector.
    """
    original_horizontal_corrector = lynx.HorizontalCorrector(
        length=jnp.array([0.2]), angle=jnp.array([4.2])
    )
    split_horizontal_corrector = lynx.Segment(
        original_horizontal_corrector.split(resolution=0.01)
    )

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_horizontal_corrector.track(incoming_beam)
    outgoing_beam_split = split_horizontal_corrector.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


@pytest.mark.xfail  # TODO: Fix this
def test_vertical_corrector_end():
    """
    Test that at the end of a split vertical corrector the result is the same as at
    the end of the original vertical corrector.
    """
    original_vertical_corrector = lynx.VerticalCorrector(
        length=jnp.array([0.2]), angle=jnp.array([4.2])
    )
    split_vertical_corrector = lynx.Segment(
        original_vertical_corrector.split(resolution=0.01)
    )

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_vertical_corrector.track(incoming_beam)
    outgoing_beam_split = split_vertical_corrector.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)
