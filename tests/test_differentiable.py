import jax.numpy as jnp

import lynx

from .resources import ARESlatticeStage3v1_9 as ares


def test_simple_quadrupole():
    """
    Simple test on a [D, Q, D] lattice with the qudrupole's k1 requiring grad, checking
    if PyTorch tracked a grad_fn into the outgoing beam.
    """
    segment = lynx.Segment(
        [
            lynx.Drift(length=jnp.array([1.0])),
            lynx.Quadrupole(
                length=jnp.array([0.2]), k1=jnp.Array([3.142]), name="my_quad"
            ),
            lynx.Drift(length=jnp.array([1.0])),
        ]
    )
    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam = segment.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None


def test_ea_magnets():
    """
    Test that gradients are tracking when the magnet settings in the ARES experimental
    area require grad.
    """
    ea = lynx.Segment.from_ocelot(ares.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    ea.AREAMQZM2.k1 = nn.Parameter(ea.AREAMQZM2.k1)
    ea.AREAMQZM1.k1 = nn.Parameter(ea.AREAMQZM1.k1)
    ea.AREAMCVM1.angle = nn.Parameter(ea.AREAMCVM1.angle)
    ea.AREAMQZM3.k1 = nn.Parameter(ea.AREAMQZM3.k1)
    ea.AREAMCHM1.angle = nn.Parameter(ea.AREAMCHM1.angle)

    outgoing_beam = ea.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None


def test_ea_incoming_parameter_beam():
    """
    Test that gradients are tracking when incoming beam (being a `ParameterBeam`)
    requires grad.
    """
    ea = lynx.Segment.from_ocelot(ares.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    incoming_beam = lynx.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    incoming_beam._mu = nn.Parameter(incoming_beam._mu)
    incoming_beam._cov = nn.Parameter(incoming_beam._cov)

    outgoing_beam = ea.track(incoming_beam)

    assert outgoing_beam._mu.grad_fn is not None
    assert outgoing_beam._cov.grad_fn is not None


def test_ea_incoming_particle_beam():
    """
    Test that gradients are tracking when incoming beam (being a `ParticleBeam`)
    requires grad.
    """
    ea = lynx.Segment.from_ocelot(ares.cell, warnings=False).subcell(
        "AREASOLA1", "AREABSCR1"
    )
    incoming_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    incoming_beam.particles = nn.Parameter(incoming_beam.particles)

    outgoing_beam = ea.track(incoming_beam)

    assert outgoing_beam.particles.grad_fn is not None
