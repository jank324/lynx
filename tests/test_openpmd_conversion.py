import jax.numpy as jnp

import lynx


def test_particlebeam_to_and_from_particlegroup():
    """
    Test that a `ParticleBeam` can be converted to an OpenPMD `ParticleGroup` and back,
    checking that the loaded `ParticleBeam` is the same as the original.
    """
    reference_energy = jnp.asarray(1e6)

    original_lynx_beam = lynx.ParticleBeam.from_parameters(
        num_particles=10_000,
        mu_x=jnp.asarray(1e-4),
        sigma_x=jnp.asarray(2e-5),
        mu_y=jnp.asarray(1e-4),
        sigma_y=jnp.asarray(2e-5),
        sigma_p=jnp.asarray(1e-4),
        energy=reference_energy,
        total_charge=jnp.asarray(1e-9),
        dtype=jnp.float64,
    )
    openpmd_particle_group = original_lynx_beam.to_openpmd_particlegroup()
    loaded_lynx_beam = lynx.ParticleBeam.from_openpmd_particlegroup(
        openpmd_particle_group, energy=reference_energy, dtype=jnp.float64
    )

    assert original_lynx_beam.num_particles == loaded_lynx_beam.num_particles
    assert jnp.allclose(original_lynx_beam.particles, loaded_lynx_beam.particles)
    assert jnp.allclose(
        original_lynx_beam.particle_charges, loaded_lynx_beam.particle_charges
    )


def test_particlebeam_to_and_from_openpmd_h5(tmp_path):
    """
    Test that a `ParticleBeam` can be saved to an OpenPMD HDF5 file and loaded back,
    checking that the loaded `ParticleBeam` is the same as the original.
    """
    reference_energy = jnp.asarray(1e6)

    original_lynx_beam = lynx.ParticleBeam.from_parameters(
        num_particles=10_000,
        mu_x=jnp.asarray(1e-4),
        sigma_x=jnp.asarray(2e-5),
        mu_y=jnp.asarray(1e-4),
        sigma_y=jnp.asarray(2e-5),
        sigma_p=jnp.asarray(1e-4),
        energy=reference_energy,
        total_charge=jnp.asarray(1e-9),
        dtype=jnp.float64,
    )
    original_lynx_beam.save_as_openpmd_h5(tmp_path / "particlegroup.h5")
    loaded_lynx_beam = lynx.ParticleBeam.from_openpmd_file(
        tmp_path / "particlegroup.h5", energy=reference_energy, dtype=jnp.float64
    )

    assert original_lynx_beam.num_particles == loaded_lynx_beam.num_particles
    assert jnp.allclose(original_lynx_beam.particles, loaded_lynx_beam.particles)
    assert jnp.allclose(
        original_lynx_beam.particle_charges, loaded_lynx_beam.particle_charges
    )
