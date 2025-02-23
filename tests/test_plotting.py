import jax.numpy as jnp

import lynx

from .resources import ARESlatticeStage3v1_9 as ares


def test_twiss_plot():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an
    example.
    """
    cell = lynx.converters.ocelot.subcell_of_ocelot(ares.cell, "AREASOLA1", "AREABSCR1")
    ares.areamqzm1.k1 = 5.0
    ares.areamqzm2.k1 = -5.0
    ares.areamcvm1.k1 = 1e-3
    ares.areamqzm3.k1 = 5.0
    ares.areamchm1.k1 = -2e-3

    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    segment = lynx.Segment.from_ocelot(cell)

    # Run the plotting to see if it raises an exception
    segment.plot_twiss(incoming_beam)


def test_reference_particle_plot():
    """
    Test that the reference particle plot does not raise an exception using the example
    from the `simple.ipynb` example notebook from the documentation.
    """
    segment = lynx.Segment(
        elements=[
            lynx.BPM(name="BPM1SMATCH"),
            lynx.Drift(length=jnp.asarray(1.0)),
            lynx.BPM(name="BPM6SMATCH"),
            lynx.Drift(length=jnp.asarray(1.0)),
            lynx.VerticalCorrector(length=jnp.asarray(0.3), name="V7SMATCH"),
            lynx.Drift(length=jnp.asarray(0.2)),
            lynx.HorizontalCorrector(length=jnp.asarray(0.3), name="H10SMATCH"),
            lynx.Drift(length=jnp.asarray(7.0)),
            lynx.HorizontalCorrector(length=jnp.asarray(0.3), name="H12SMATCH"),
            lynx.Drift(length=jnp.asarray(0.05)),
            lynx.BPM(name="BPM13SMATCH"),
        ]
    )

    segment.V7SMATCH.angle = jnp.asarray(3.142e-3)

    incoming = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    segment.plot_overview(incoming=incoming)


def test_twiss_plot_vectorized_2d():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an
    example and when the model has two vector dimensions.
    """
    segment = lynx.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    segment.AREAMQZM1.k1 = jnp.asarray(5.0)
    segment.AREAMQZM2.k1 = jnp.asarray([[-5.0, -2.0, -1.0], [1.0, 2.0, 5.0]])
    segment.AREAMCVM1.k1 = jnp.asarray(1e-3)
    segment.AREAMQZM3.k1 = jnp.asarray(5.0)
    segment.AREAMCHM1.k1 = jnp.asarray(-2e-3)
    segment.Drift_AREAMCHM1.length = (
        jnp.FloatTensor(2, 3).uniform_(0.9, 1.1) * segment.Drift_AREAMCHM1.length
    )

    incoming = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    segment.plot_twiss(incoming=incoming, vector_idx=(0, 2))


def test_reference_particle_plot_vectorized_2d():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an
    example and when the model has two vector dimensions.
    """
    segment = lynx.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    segment.AREAMQZM1.k1 = jnp.asarray(5.0)
    segment.AREAMQZM2.k1 = jnp.asarray([[-5.0, -2.0, -1.0], [1.0, 2.0, 5.0]])
    segment.AREAMCVM1.k1 = jnp.asarray(1e-3)
    segment.AREAMQZM3.k1 = jnp.asarray(5.0)
    segment.AREAMCHM1.k1 = jnp.asarray(-2e-3)
    segment.Drift_AREAMCHM1.length = (
        jnp.FloatTensor(2, 3).uniform_(0.9, 1.1) * segment.Drift_AREAMCHM1.length
    )

    incoming = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    segment.plot_overview(incoming=incoming, resolution=0.1, vector_idx=(0, 2))


def test_plotting_with_nonleaf_tensors():
    """Test that the plotting routines can handle elements with non-leaf tensors."""
    segment = lynx.Segment(
        elements=[
            lynx.Drift(length=jnp.asarray(1.0, requires_grad=True)),
            lynx.BPM(is_active=True),
        ]
    )

    incoming = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Prepopulate the segment
    segment.track(incoming)

    # Test that plotting does not raise an exception
    segment.plot_overview(incoming=incoming)
    segment.plot_twiss(incoming=incoming)


def test_plotting_with_gradients():
    """
    Test that plotting doesn't raise an exception for segments that contain tensors
    that require gradients.
    """
    segment = lynx.Segment(
        elements=[lynx.Drift(length=jnp.asarray(1.0, requires_grad=True))]
    )
    beam = lynx.ParameterBeam.from_parameters()

    segment.plot_overview(incoming=beam)
    segment.plot_twiss(incoming=beam)


def test_plot_6d_particle_beam_distribution():
    """Test that the 6D `ParticleBeam` distribution plot does not raise an exception."""
    beam = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    _ = beam.plot_distribution(bin_ranges="unit_same", plot_2d_kws={"contour": True})


def test_plot_particle_beam_point_cloud():
    """Test that the `ParticleBeam`'s point cloud plot does not raise an exception."""
    beam = lynx.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    _ = beam.plot_point_cloud()
