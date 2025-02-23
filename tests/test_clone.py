import jax.numpy as jnp
import pytest

import lynx


@pytest.mark.parametrize(
    "ElementClass",
    [
        lynx.Cavity,
        lynx.Dipole,
        lynx.Drift,
        lynx.HorizontalCorrector,
        lynx.Quadrupole,
        lynx.RBend,
        lynx.Solenoid,
        lynx.TransverseDeflectingCavity,
        lynx.Undulator,
        lynx.VerticalCorrector,
    ],
)
def test_element_buffer_contents_and_location(ElementClass):
    """
    Test that the buffers of cloned elements have the same content while not sharing the
    same memory location.
    """
    element = ElementClass(length=jnp.asarray(1.0))
    clone = element.clone()

    for buffer, buffer_clone in zip(element.buffers(), clone.buffers()):
        assert jnp.allclose(buffer, buffer_clone)
        assert not buffer.data_ptr() == buffer_clone.data_ptr()


@pytest.mark.parametrize("BeamClass", [lynx.ParameterBeam, lynx.ParticleBeam])
def test_beam_buffer_contents_and_location(BeamClass):
    """
    Test that the buffers of cloned beams have the same content while not sharing the
    same memory location.
    """
    beam = BeamClass.from_parameters()
    clone = beam.clone()

    for buffer, buffer_clone in zip(beam.buffers(), clone.buffers()):
        assert jnp.allclose(buffer, buffer_clone)
        assert not buffer.data_ptr() == buffer_clone.data_ptr()
