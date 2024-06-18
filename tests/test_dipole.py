import jax.numpy as jnp

from lynx import Dipole, Drift, ParticleBeam, Segment


def test_dipole_off():
    """
    Test that a dipole with angle=0 behaves still like a drift.
    """
    dipole = Dipole(length=jnp.array([1.0]), angle=jnp.array([0.0]))
    drift = Drift(length=jnp.array([1.0]))
    incoming_beam = ParticleBeam.from_parameters(
        sigma_xp=jnp.array([2e-7]), sigma_yp=jnp.array([2e-7])
    )
    outbeam_dipole_off = dipole(incoming_beam)
    outbeam_drift = drift(incoming_beam)

    dipole.angle = jnp.array([1.0], device=dipole.angle.device)
    outbeam_dipole_on = dipole(incoming_beam)

    assert jnp.allclose(outbeam_dipole_off.sigma_x, outbeam_drift.sigma_x)
    assert not jnp.allclose(outbeam_dipole_on.sigma_x, outbeam_drift.sigma_x)


def test_dipole_batched_execution():
    """
    Test that a dipole with batch dimensions behaves as expected.
    """
    batch_shape = (3,)
    incoming = ParticleBeam.from_parameters(
        num_particles=1_000_000, energy=jnp.array([1e9]), mu_x=jnp.array([1e-5])
    ).broadcast(batch_shape)
    segment = Segment(
        [
            Dipole(length=jnp.array([0.5, 0.5, 0.5]), angle=jnp.array([0.1, 0.2, 0.1])),
            Drift(length=jnp.array([0.5])).broadcast(batch_shape),
        ]
    )
    outgoing = segment(incoming)

    # Check that dipole with same bend angle produce same output
    assert jnp.allclose(outgoing.particles[0], outgoing.particles[2])

    # Check different angles do make a difference
    assert not jnp.allclose(outgoing.particles[0], outgoing.particles[1])
