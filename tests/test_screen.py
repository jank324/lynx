import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lynx

from .resources import ARESlatticeStage3v1_9 as ocelot_lattice


@pytest.mark.parametrize("screen_method", ["histogram", "kde"])
def test_reading_shows_beam_particle(screen_method):
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0)),
            lynx.Screen(
                resolution=(100, 100),
                pixel_size=jnp.asarray((1e-5, 1e-5)),
                is_active=True,
                method=screen_method,
                name="my_screen",
            ),
        ],
    )
    beam = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, jnp.Array)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, jnp.Array)
    assert segment.my_screen.reading.shape == (100, 100)
    assert jnp.all(segment.my_screen.reading >= 0.0)
    assert jnp.any(segment.my_screen.reading > 0.0)


@pytest.mark.parametrize("kde_bandwidth", [5e-6, 1e-5, 5e-5])
def test_screen_kde_bandwidth(kde_bandwidth):
    """Test screen reading with KDE method and different explicit bandwidths."""

    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0)),
            lynx.Screen(
                resolution=(100, 100),
                pixel_size=jnp.asarray((1e-5, 1e-5)),
                is_active=True,
                method="kde",
                name="my_screen",
                kde_bandwidth=jnp.asarray(kde_bandwidth),
            ),
        ],
    )
    beam = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, jnp.Array)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, jnp.Array)
    assert segment.my_screen.reading.shape == (100, 100)
    assert jnp.all(segment.my_screen.reading >= 0.0)
    assert jnp.any(segment.my_screen.reading > 0.0)


@pytest.mark.parametrize("screen_method", ["histogram", "kde"])
def test_reading_shows_beam_parameter(screen_method):
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0)),
            lynx.Screen(
                resolution=(100, 100),
                pixel_size=jnp.asarray((1e-5, 1e-5)),
                is_active=True,
                method=screen_method,
                name="my_screen",
            ),
        ],
        name="my_segment",
    )
    beam = lynx.ParameterBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, jnp.Array)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, jnp.Array)
    assert segment.my_screen.reading.shape == (100, 100)
    assert jnp.all(segment.my_screen.reading >= 0.0)
    assert jnp.any(segment.my_screen.reading > 0.0)


@pytest.mark.parametrize("screen_method", ["histogram", "kde"])
def test_reading_shows_beam_ares(screen_method):
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = lynx.Segment.from_ocelot(ocelot_lattice.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    beam = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    segment.AREABSCR1.method = screen_method

    segment.AREABSCR1.resolution = (2448, 2040)
    segment.AREABSCR1.pixel_size = jnp.asarray(
        (3.3198e-6, 2.4469e-6),
        device=segment.AREABSCR1.pixel_size.device,
        dtype=segment.AREABSCR1.pixel_size.dtype,
    )
    segment.AREABSCR1.binning = 1
    segment.AREABSCR1.is_active = True

    assert isinstance(segment.AREABSCR1.reading, jnp.Array)
    assert segment.AREABSCR1.reading.shape == (2040, 2448)
    assert np.allclose(segment.AREABSCR1.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.AREABSCR1.reading, jnp.Array)
    assert segment.AREABSCR1.reading.shape == (2040, 2448)
    assert jnp.all(segment.AREABSCR1.reading >= 0.0)
    assert jnp.any(segment.AREABSCR1.reading > 0.0)


def test_reading_dtype_conversion():
    """Test that a dtype conversion is correctly reflected in the screen reading."""
    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0), dtype=jnp.float32),
            lynx.Screen(name="screen", is_active=True, dtype=jnp.float32),
        ],
    )
    beam = lynx.ParameterBeam.from_parameters(dtype=jnp.float32)
    assert segment.screen.reading.dtype == jnp.float32

    # Test generating new image
    cloned = segment.clone()
    cloned.track(beam)
    cloned = cloned.double()
    assert jnp.all(jnp.isnan(cloned.screen.cached_reading))
    assert cloned.screen.reading.dtype == jnp.float64

    # Test reading from cache
    segment.track(beam)
    assert segment.screen.reading.dtype == jnp.float32
    assert segment.screen.cached_reading.dtype == jnp.float32
    segment = segment.double()
    assert segment.screen.cached_reading.dtype == jnp.float64
    assert segment.screen.reading.dtype == jnp.float64
