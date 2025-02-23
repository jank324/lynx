import jax.numpy as jnp
import pytest

import lynx


@pytest.mark.parametrize("is_bpm_active", [True, False])
@pytest.mark.parametrize("beam_class", [lynx.ParticleBeam, lynx.ParameterBeam])
def test_no_tracking_error(is_bpm_active, beam_class):
    """Test that tracking a beam through an inactive BPM does not raise an error."""
    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0)),
            lynx.BPM(name="my_bpm"),
            lynx.Drift(length=jnp.asarray(1.0)),
        ],
    )
    beam = beam_class.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    segment.my_bpm.is_active = is_bpm_active

    _ = segment.track(beam)


def test_reading_dtype_conversion():
    """Test that a dtype conversion is correctly reflected in the BPM reading."""
    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0), dtype=jnp.float32),
            lynx.BPM(name="bpm", is_active=True, dtype=jnp.float32),
        ],
    )
    beam = lynx.ParameterBeam.from_parameters(dtype=jnp.float32)
    assert segment.bpm.reading.dtype == jnp.float32

    segment.track(beam)
    assert segment.bpm.reading.dtype == jnp.float32

    segment = segment.double()
    assert segment.bpm.reading.dtype == jnp.float64
