import jax.numpy as jnp
import pytest

import lynx


def test_assert_ei_greater_zero():
    """
    Reproduces

    ```
       1127 Ef = (energy + delta_energy) / electron_mass_eV
       1128 Ep = (Ef - Ei) / self.length  # Derivative of the energy
    -> 1129 assert Ei > 0, "Initial energy must be larger than 0"
       1131 alpha = jnp.sqrt(eta / 8) / jnp.cos(phi) * jnp.log(Ef / Ei)
       1133 r11 = jnp.cos(alpha) - jnp.sqrt(2 / eta) * jnp.cos(phi) * jnp.sin(alpha)   # noqa: E501

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    ```
    """
    cavity = lynx.Cavity(
        length=jnp.array([3.0441, 3.0441, 3.0441]),
        voltage=jnp.array([48198468.0, 48198468.0, 48198468.0]),
        phase=jnp.array([48198468.0, 48198468.0, 48198468.0]),
        frequency=jnp.array([2.8560e09, 2.8560e09, 2.8560e09]),
        name="k26_2a",
    )
    beam = lynx.ParticleBeam.from_parameters(
        num_particles=100_000, sigma_x=jnp.asarray(1e-5)
    )

    _ = cavity.track(beam)


@pytest.mark.parametrize(
    "voltage",
    [jnp.asarray([0.0, 0.0]), jnp.asarray([0.0, 1e6]), jnp.asarray([1e6, 1e6])],
)
def test_vectorized_cavity_zero_voltage(voltage):
    """
    Tests that a vectorised cavity with zero voltage does not produce NaNs and that
    zero voltage can be vectorised with non-zero voltage.

    This was a bug introduced during the vectorisation of Cheetah, when the special
    case of zero was removed and the `_cavity_rmatrix` method was also used in the case
    of zero voltage. The latter produced NaNs in the transfer matrix when the voltage
    is zero.
    """
    cavity = lynx.Cavity(
        length=jnp.asarray([3.0441, 3.0441]),
        voltage=voltage,
        phase=jnp.asarray([-0.0, -0.0]),
        frequency=jnp.asarray([2.8560e09, 2.8560e09]),
        name="k27_1a",
        dtype=jnp.float64,
    )
    incoming = lynx.ParameterBeam.from_parameters(
        mu_x=jnp.asarray(0.0),
        mu_px=jnp.asarray(0.0),
        mu_y=jnp.asarray(0.0),
        mu_py=jnp.asarray(0.0),
        sigma_x=jnp.asarray(4.8492e-06),
        sigma_px=jnp.asarray(1.5603e-07),
        sigma_y=jnp.asarray(4.1209e-07),
        sigma_py=jnp.asarray(1.1035e-08),
        sigma_tau=jnp.asarray(1.0000e-10),
        sigma_p=jnp.asarray(1.0000e-06),
        energy=jnp.asarray(8.0000e09),
        total_charge=jnp.asarray(0.0),
        dtype=jnp.float64,
    )

    outgoing = cavity.track(incoming)

    assert not jnp.isnan(cavity.transfer_map(incoming.energy)).any()

    assert not jnp.isnan(outgoing.sigma_x).any()
    assert not jnp.isnan(outgoing.sigma_y).any()
    assert not jnp.isnan(outgoing.beta_x).any()
    assert not jnp.isnan(outgoing.beta_y).any()
