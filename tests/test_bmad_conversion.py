import jax.numpy as jnp

import lynx


def test_bmad_tutorial():
    """Test importing the lattice example file from the Bmad and Tao tutorial."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"
    converted = lynx.Segment.from_bmad(file_path)
    converted.name = "bmad_tutorial"

    correct = lynx.Segment(
        [
            lynx.Drift(length=jnp.array([0.5]), name="d"),
            lynx.Dipole(
                length=jnp.array([0.5]), e1=jnp.array([0.1]), name="b"
            ),  # TODO: What are g and dg?
            lynx.Quadrupole(length=jnp.array([0.6]), k1=jnp.array([0.23]), name="q"),
        ],
        name="bmad_tutorial",
    )

    assert converted.name == correct.name
    assert [element.name for element in converted.elements] == [
        element.name for element in correct.elements
    ]
    assert converted.d.length == correct.d.length
    assert converted.b.length == correct.b.length
    assert converted.b.e1 == correct.b.e1
    assert converted.q.length == correct.q.length
    assert converted.q.k1 == correct.q.k1
