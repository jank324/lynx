"""
Tests that ensure that both beam types produce (roughly) the same results.
"""

import jax.numpy as jnp

import lynx


def test_from_twiss():
    """
    Test that a beams created from Twiss parameters have the same properties.
    """
    parameter_beam = lynx.ParameterBeam.from_twiss(
        beta_x=jnp.asarray(5.91253676811640894),
        alpha_x=jnp.asarray(3.55631307633660354),
        emittance_x=jnp.asarray(3.494768647122823e-09),
        beta_y=jnp.asarray(5.91253676811640982),
        alpha_y=jnp.asarray(2e-7),
        emittance_y=jnp.asarray(3.497810737006068e-09),
        energy=jnp.asarray(6e6),
    )
    particle_beam = lynx.ParticleBeam.from_twiss(
        num_particles=jnp.array(
            [10_000_000]
        ),  # Large number of particles reduces noise
        beta_x=jnp.asarray(5.91253676811640894),
        alpha_x=jnp.asarray(3.55631307633660354),
        emittance_x=jnp.asarray(3.494768647122823e-09),
        beta_y=jnp.asarray(5.91253676811640982),
        alpha_y=jnp.asarray(2e-7),
        emittance_y=jnp.asarray(3.497810737006068e-09),
        energy=jnp.asarray(6e6),
    )

    assert jnp.isclose(parameter_beam.mu_x, particle_beam.mu_x, atol=1e-6)
    assert jnp.isclose(parameter_beam.mu_y, particle_beam.mu_y, atol=1e-6)
    assert jnp.isclose(parameter_beam.sigma_x, particle_beam.sigma_x, rtol=1e-3)
    assert jnp.isclose(parameter_beam.sigma_y, particle_beam.sigma_y, rtol=1e-3)
    assert jnp.isclose(parameter_beam.mu_px, particle_beam.mu_px, atol=1e-6)
    assert jnp.isclose(parameter_beam.mu_py, particle_beam.mu_py, atol=1e-6)
    assert jnp.isclose(parameter_beam.sigma_px, particle_beam.sigma_px, rtol=1e-3)
    assert jnp.isclose(parameter_beam.sigma_py, particle_beam.sigma_py, rtol=1e-3)
    assert jnp.isclose(parameter_beam.mu_tau, particle_beam.mu_tau)
    assert jnp.isclose(parameter_beam.sigma_tau, particle_beam.sigma_tau)
    assert jnp.isclose(parameter_beam.mu_p, particle_beam.mu_p)
    assert jnp.isclose(parameter_beam.sigma_p, particle_beam.sigma_p)


def test_drift():
    """Test that the drift output for both beam types is roughly the same."""

    # Set up lattice
    lynx_drift = lynx.Drift(length=jnp.asarray(1.0))

    # Parameter beam
    incoming_parameter_beam = lynx.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    outgoing_parameter_beam = lynx_drift.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    outgoing_particle_beam = lynx_drift.track(incoming_particle_beam)

    # Compare
    assert jnp.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert jnp.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_px, outgoing_particle_beam.mu_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_py, outgoing_particle_beam.mu_py, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_px, outgoing_particle_beam.sigma_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_py, outgoing_particle_beam.sigma_py, rtol=1e-2
    )


def test_quadrupole():
    """Test that the quadrupole output for both beam types is roughly the same."""

    # Set up lattice
    lynx_quadrupole = lynx.Quadrupole(length=jnp.asarray(0.15), k1=jnp.asarray(4.2))

    # Parameter beam
    incoming_parameter_beam = lynx.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    outgoing_parameter_beam = lynx_quadrupole.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    outgoing_particle_beam = lynx_quadrupole.track(incoming_particle_beam)

    # Compare
    assert jnp.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert jnp.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_px, outgoing_particle_beam.mu_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_py, outgoing_particle_beam.mu_py, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_px, outgoing_particle_beam.sigma_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_py, outgoing_particle_beam.sigma_py, rtol=1e-2
    )


def test_cavity_from_astra():
    """
    Test that the cavity output for both beam types is roughly the same. This test uses
    a beam converted from an ASTRA beam file.
    """

    # Set up lattice
    lynx_cavity = lynx.Cavity(
        length=jnp.asarray(1.0377),
        voltage=jnp.asarray(0.01815975e9),
        frequency=jnp.asarray(1.3e9),
        phase=jnp.asarray(0.0),
    )

    # Parameter beam
    incoming_parameter_beam = lynx.ParameterBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    outgoing_parameter_beam = lynx_cavity.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = lynx.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    outgoing_particle_beam = lynx_cavity.track(incoming_particle_beam)

    # Compare
    assert jnp.isclose(
        outgoing_parameter_beam.beta_x, outgoing_particle_beam.beta_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.alpha_x, outgoing_particle_beam.alpha_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.beta_y, outgoing_particle_beam.beta_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.alpha_y, outgoing_particle_beam.alpha_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.emittance_x, outgoing_particle_beam.emittance_x
    )
    assert jnp.isclose(
        outgoing_parameter_beam.emittance_y, outgoing_particle_beam.emittance_y
    )
    assert jnp.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert jnp.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_px, outgoing_particle_beam.mu_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_py, outgoing_particle_beam.mu_py, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_px, outgoing_particle_beam.sigma_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_py, outgoing_particle_beam.sigma_py, rtol=1e-2
    )


def test_cavity_from_twiss():
    """
    Test that the cavity output for both beam types is roughly the same. This test uses
    a beam generated from Twiss parameters.
    """

    # Set up lattice
    lynx_cavity = lynx.Cavity(
        length=jnp.asarray(1.0377),
        voltage=jnp.asarray(0.01815975e9),
        frequency=jnp.asarray(1.3e9),
        phase=jnp.asarray(0.0),
    )

    # Parameter beam
    incoming_parameter_beam = lynx.ParameterBeam.from_twiss(
        beta_x=jnp.asarray(5.91253677),
        alpha_x=jnp.asarray(3.55631308),
        beta_y=jnp.asarray(5.91253677),
        alpha_y=jnp.asarray(3.55631308),
        emittance_x=jnp.asarray(3.494768647122823e-09),
        emittance_y=jnp.asarray(3.497810737006068e-09),
        energy=jnp.asarray(6e6),
    )
    outgoing_parameter_beam = lynx_cavity.track(incoming_parameter_beam)

    # Particle beam
    incoming_particle_beam = lynx.ParticleBeam.from_twiss(
        num_particles=1_000_000,
        beta_x=jnp.asarray(5.91253677),
        alpha_x=jnp.asarray(3.55631308),
        beta_y=jnp.asarray(5.91253677),
        alpha_y=jnp.asarray(3.55631308),
        emittance_x=jnp.asarray(3.494768647122823e-09),
        emittance_y=jnp.asarray(3.497810737006068e-09),
        energy=jnp.asarray(6e6),
    )
    outgoing_particle_beam = lynx_cavity.track(incoming_particle_beam)

    # Compare
    assert jnp.isclose(
        outgoing_parameter_beam.beta_x, outgoing_particle_beam.beta_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.alpha_x, outgoing_particle_beam.alpha_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.beta_y, outgoing_particle_beam.beta_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.alpha_y, outgoing_particle_beam.alpha_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.emittance_x, outgoing_particle_beam.emittance_x
    )
    assert jnp.isclose(
        outgoing_parameter_beam.emittance_y, outgoing_particle_beam.emittance_y
    )
    assert jnp.isclose(outgoing_parameter_beam.energy, outgoing_particle_beam.energy)
    assert jnp.isclose(
        outgoing_parameter_beam.mu_x, outgoing_particle_beam.mu_x, atol=1e-6
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_y, outgoing_particle_beam.mu_y, atol=1e-6
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_x, outgoing_particle_beam.sigma_x, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_y, outgoing_particle_beam.sigma_y, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_px, outgoing_particle_beam.mu_px, atol=1e-6
    )
    assert jnp.isclose(
        outgoing_parameter_beam.mu_py, outgoing_particle_beam.mu_py, atol=1e-6
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_px, outgoing_particle_beam.sigma_px, rtol=1e-2
    )
    assert jnp.isclose(
        outgoing_parameter_beam.sigma_py, outgoing_particle_beam.sigma_py, rtol=1e-2
    )
