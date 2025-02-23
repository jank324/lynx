import jax.numpy as jnp
import pytest

import lynx
from lynx.utils import is_mps_available_and_functional


def test_bmad_tutorial():
    """Test importing the lattice example file from the Bmad and Tao tutorial."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"
    converted = lynx.Segment.from_bmad(file_path)
    converted.name = "bmad_tutorial"

    correct = lynx.Segment(
        [
            lynx.Drift(length=jnp.asarray([0.5]), name="d"),
            lynx.Dipole(
                length=jnp.asarray([0.5]), dipole_e1=jnp.asarray([0.1]), name="b"
            ),  # TODO: What are g and dg?
            lynx.Quadrupole(
                length=jnp.asarray([0.6]), k1=jnp.asarray([0.23]), name="q"
            ),
        ],
        name="bmad_tutorial",
    )

    assert converted.name == correct.name
    assert converted.length == correct.length
    assert [element.name for element in converted.elements] == [
        element.name for element in correct.elements
    ]
    assert converted.d.length == correct.d.length
    assert converted.b.length == correct.b.length
    assert converted.b.dipole_e1 == correct.b.dipole_e1
    assert converted.q.length == correct.q.length
    assert converted.q.k1 == correct.q.k1


@pytest.mark.parametrize(
    "device",
    [
        jnp.device("cpu"),
        pytest.param(
            jnp.device("cuda"),
            marks=pytest.mark.skipif(
                not jnp.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            jnp.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_device_passing(device: jnp.device):
    """Test that the device is passed correctly."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the device
    converted = lynx.Segment.from_bmad(file_path, device=device)

    # Check that the properties of the loaded elements are on the correct device
    assert converted.d.length.device.type == device.type
    assert converted.b.length.device.type == device.type
    assert converted.b.dipole_e1.device.type == device.type
    assert converted.q.length.device.type == device.type
    assert converted.q.k1.device.type == device.type


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtype_passing(dtype: jnp.dtype):
    """Test that the dtype is passed correctly."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the dtype
    converted = lynx.Segment.from_bmad(file_path, dtype=dtype)

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.d.length.dtype == dtype
    assert converted.b.length.dtype == dtype
    assert converted.b.dipole_e1.dtype == dtype
    assert converted.q.length.dtype == dtype
    assert converted.q.k1.dtype == dtype
