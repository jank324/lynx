import jax.numpy as jnp
import numpy as np

from lynx import ParticleBeam


def test_create_from_parameters():
    """
    Test that a `ParticleBeam` created from parameters actually has those parameters.
    """
    beam = ParticleBeam.from_parameters(
        num_particles=torch.tensor([1_000_000]),
        mu_x=jnp.array([1e-5]),
        mu_xp=jnp.array([1e-7]),
        mu_y=jnp.array([2e-5]),
        mu_yp=jnp.array([2e-7]),
        sigma_x=jnp.array([1.75e-7]),
        sigma_xp=jnp.array([2e-7]),
        sigma_y=jnp.array([1.75e-7]),
        sigma_yp=jnp.array([2e-7]),
        sigma_s=jnp.array([0.000001]),
        sigma_p=jnp.array([0.000001]),
        cor_x=jnp.array([0.0]),
        cor_y=jnp.array([0.0]),
        cor_s=jnp.array([0.0]),
        energy=jnp.array([1e7]),
        total_charge=jnp.array([1e-9]),
    )

    assert beam.num_particles == 1_000_000
    assert np.isclose(beam.mu_x.cpu().numpy(), 1e-5)
    assert np.isclose(beam.mu_xp.cpu().numpy(), 1e-7)
    assert np.isclose(beam.mu_y.cpu().numpy(), 2e-5)
    assert np.isclose(beam.mu_yp.cpu().numpy(), 2e-7)
    assert np.isclose(beam.sigma_x.cpu().numpy(), 1.75e-7)
    assert np.isclose(beam.sigma_xp.cpu().numpy(), 2e-7)
    assert np.isclose(beam.sigma_y.cpu().numpy(), 1.75e-7)
    assert np.isclose(beam.sigma_yp.cpu().numpy(), 2e-7)
    assert np.isclose(beam.sigma_s.cpu().numpy(), 0.000001)
    assert np.isclose(beam.sigma_p.cpu().numpy(), 0.000001)
    assert np.isclose(beam.energy.cpu().numpy(), 1e7)
    assert np.isclose(beam.total_charge.cpu().numpy(), 1e-9)


def test_transform_to():
    """
    Test that a `ParticleBeam` transformed to new parameters actually has those new
    parameters.
    """
    original_beam = ParticleBeam.from_parameters()
    transformed_beam = original_beam.transformed_to(
        mu_x=jnp.array([1e-5]),
        mu_xp=jnp.array([1e-7]),
        mu_y=jnp.array([2e-5]),
        mu_yp=jnp.array([2e-7]),
        sigma_x=jnp.array([1.75e-7]),
        sigma_xp=jnp.array([2e-7]),
        sigma_y=jnp.array([1.75e-7]),
        sigma_yp=jnp.array([2e-7]),
        sigma_s=jnp.array([0.000001]),
        sigma_p=jnp.array([0.000001]),
        energy=jnp.array([1e7]),
        total_charge=jnp.array([1e-9]),
    )

    assert isinstance(transformed_beam, ParticleBeam)
    assert original_beam.num_particles == transformed_beam.num_particles

    assert np.isclose(transformed_beam.mu_x.cpu().numpy(), 1e-5)
    assert np.isclose(transformed_beam.mu_xp.cpu().numpy(), 1e-7)
    assert np.isclose(transformed_beam.mu_y.cpu().numpy(), 2e-5)
    assert np.isclose(transformed_beam.mu_yp.cpu().numpy(), 2e-7)
    assert np.isclose(transformed_beam.sigma_x.cpu().numpy(), 1.75e-7)
    assert np.isclose(transformed_beam.sigma_xp.cpu().numpy(), 2e-7)
    assert np.isclose(transformed_beam.sigma_y.cpu().numpy(), 1.75e-7)
    assert np.isclose(transformed_beam.sigma_yp.cpu().numpy(), 2e-7)
    assert np.isclose(transformed_beam.sigma_s.cpu().numpy(), 0.000001)
    assert np.isclose(transformed_beam.sigma_p.cpu().numpy(), 0.000001)
    assert np.isclose(transformed_beam.energy.cpu().numpy(), 1e7)
    assert np.isclose(transformed_beam.total_charge.cpu().numpy(), 1e-9)


def test_from_twiss_to_twiss():
    """
    Test that a `ParameterBeam` created from twiss parameters actually has those
    parameters.
    """
    beam = ParticleBeam.from_twiss(
        num_particles=torch.tensor([10_000_000]),
        beta_x=jnp.array([5.91253676811640894]),
        alpha_x=jnp.array([3.55631307633660354]),
        emittance_x=jnp.array([3.494768647122823e-09]),
        beta_y=jnp.array([5.91253676811640982]),
        alpha_y=jnp.array([1.0]),  # TODO: set realistic value
        emittance_y=jnp.array([3.497810737006068e-09]),
        energy=jnp.array([6e6]),
    )
    # rather loose rtol is needed here due to the random sampling of the beam
    assert np.isclose(beam.beta_x.cpu().numpy(), 5.91253676811640894, rtol=1e-2)
    assert np.isclose(beam.alpha_x.cpu().numpy(), 3.55631307633660354, rtol=1e-2)
    assert np.isclose(beam.emittance_x.cpu().numpy(), 3.494768647122823e-09, rtol=1e-2)
    assert np.isclose(beam.beta_y.cpu().numpy(), 5.91253676811640982, rtol=1e-2)
    assert np.isclose(beam.alpha_y.cpu().numpy(), 1.0, rtol=1e-2)
    assert np.isclose(beam.emittance_y.cpu().numpy(), 3.497810737006068e-09, rtol=1e-2)
    assert np.isclose(beam.energy.cpu().numpy(), 6e6)


def test_generate_uniform_ellipsoid_batched():
    """
    Test that a `ParticleBeam` generated from a uniform 3D ellipsoid has the correct
    parameters, i.e. the all particles are within the ellipsoid, and that the other
    beam parameters are as they would be for a Gaussian beam.
    """
    radius_x = jnp.array([1e-3, 2e-3])
    radius_y = jnp.array([1e-4, 2e-4])
    radius_s = jnp.array([1e-5, 2e-5])

    num_particles = torch.tensor(1_000_000)
    sigma_xp = jnp.array([2e-7, 1e-7])
    sigma_yp = jnp.array([3e-7, 2e-7])
    sigma_p = jnp.array([0.000001, 0.000002])
    energy = jnp.array([1e7, 2e7])
    total_charge = jnp.array([1e-9, 3e-9])

    num_particles = torch.tensor(1_000_000)
    beam = ParticleBeam.uniform_3d_ellipsoid(
        num_particles=num_particles,
        radius_x=radius_x,
        radius_y=radius_y,
        radius_s=radius_s,
        sigma_xp=sigma_xp,
        sigma_yp=sigma_yp,
        sigma_p=sigma_p,
        energy=energy,
        total_charge=total_charge,
    )

    assert beam.num_particles == num_particles
    assert jnp.all(beam.xs.abs().transpose(0, 1) <= radius_x)
    assert jnp.all(beam.ys.abs().transpose(0, 1) <= radius_y)
    assert jnp.all(beam.ss.abs().transpose(0, 1) <= radius_s)
    assert jnp.allclose(beam.sigma_xp, sigma_xp)
    assert jnp.allclose(beam.sigma_yp, sigma_yp)
    assert jnp.allclose(beam.sigma_p, sigma_p)
    assert jnp.allclose(beam.energy, energy)
    assert jnp.allclose(beam.total_charge, total_charge)
