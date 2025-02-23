import jax.numpy as jnp
import pytest

import lynx


def test_drift_end():
    """
    Test that at the end of a split drift the result is the same as at the end of the
    original drift.
    """
    original_drift = lynx.Drift(length=jnp.asarray([2.0, 2.5]))
    split_drift = lynx.Segment(original_drift.split(resolution=jnp.asarray(0.1)))

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original_drift.track(incoming_beam)
    outgoing_beam_split = split_drift.track(incoming_beam)

    assert jnp.allclose(outgoing_beam_original.particles, outgoing_beam_split.particles)


def test_quadrupole_end():
    """
    Test that at the end of a split quadrupole the result is the same as at the end of
    the original quadrupole.
    """
    original_quadrupole = lynx.Quadrupole(
        length=jnp.asarray([0.2, 0.3]), k1=jnp.asarray([4.2, 3.6])
    )
    split_quadrupole = lynx.Segment(
        original_quadrupole.split(resolution=jnp.asarray(0.01))
    )

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
        length=jnp.asarray([1.0377, 2.0377]),
        voltage=jnp.asarray([0.01815975e9, 9.15975e6]),
        frequency=jnp.asarray([1.3e9, 3.9e9]),
        phase=jnp.asarray([0.0, 4.2]),
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
    original_solenoid = lynx.Solenoid(
        length=jnp.asarray([0.2, 0.3]), k=jnp.asarray([4.2, 3.6])
    )
    split_solenoid = lynx.Segment(original_solenoid.split(resolution=jnp.asarray(0.01)))

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

    original_dipole = lynx.Dipole(
        length=jnp.asarray([0.2, 0.3]), angle=jnp.asarray([4.2, 3.6])
    )
    split_dipole = lynx.Segment(original_dipole.split(resolution=jnp.asarray(0.01)))

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
    original_undulator = lynx.Undulator(length=jnp.asarray([3.142, 2.7]))
    split_undulator = lynx.Segment(
        original_undulator.split(resolution=jnp.asarray(0.1))
    )

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
        length=jnp.asarray([0.2, 0.3]), angle=jnp.asarray([4.2, 3.6])
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
        length=jnp.asarray([0.2, 0.3]), angle=jnp.asarray([4.2, 3.6])
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


@pytest.mark.parametrize(
    "ElementType",
    [
        lynx.Cavity,
        lynx.Dipole,
        lynx.Drift,
        lynx.HorizontalCorrector,
        lynx.Quadrupole,
        lynx.RBend,
        lynx.Solenoid,
        lynx.Undulator,
        lynx.VerticalCorrector,
    ],
)
def test_split_preserves_dtype(ElementType):
    """
    Test that the dtype of a drift section's splits is the same as the original drift.
    """
    original = ElementType(length=jnp.asarray(2.0), dtype=jnp.float64)
    splits = original.split(resolution=jnp.asarray(0.1))

    for split in splits:
        assert original.length.dtype == split.length.dtype
