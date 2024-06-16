import jax.numpy as jnp
import numpy as np

from lynx import ParameterBeam


def test_create_from_parameters():
    """
    Test that a `ParameterBeam` created from parameters actually has those parameters.
    """
    beam = ParameterBeam.from_parameters(
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
    )

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


def test_transform_to():
    """
    Test that a `ParameterBeam` transformed to new parameters actually has those new
    parameters.
    """
    original_beam = ParameterBeam.from_parameters()
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

    assert isinstance(transformed_beam, ParameterBeam)
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
    beam = ParameterBeam.from_twiss(
        beta_x=jnp.array([5.91253676811640894]),
        alpha_x=jnp.array([3.55631307633660354]),
        emittance_x=jnp.array([3.494768647122823e-09]),
        beta_y=jnp.array([5.91253676811640982]),
        alpha_y=jnp.array([2e-7]),
        emittance_y=jnp.array([3.497810737006068e-09]),
        energy=jnp.array([6e6]),
    )

    assert np.isclose(beam.beta_x.cpu().numpy(), 5.91253676811640894)
    assert np.isclose(beam.alpha_x.cpu().numpy(), 3.55631307633660354)
    assert np.isclose(beam.emittance_x.cpu().numpy(), 3.494768647122823e-09)
    assert np.isclose(beam.beta_y.cpu().numpy(), 5.91253676811640982)
    assert np.isclose(beam.alpha_y.cpu().numpy(), 2e-7)
    assert np.isclose(beam.emittance_y.cpu().numpy(), 3.497810737006068e-09)
    assert np.isclose(beam.energy.cpu().numpy(), 6e6)
