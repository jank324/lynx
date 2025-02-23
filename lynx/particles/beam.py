from abc import ABC, abstractmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from scipy.constants import physical_constants

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Beam(ABC, nn.Module):
    r"""
    Parent class to represent a beam of particles. You should not instantiate this
    class directly, but use one of the subclasses.

    Lynx uses a 7D vector to describe the state of a particle.
    It contains the 6D phase space vector (x, px, y, yp, tau, p) and an additional
    dimension (always 1) for convenient calculations.

    The phase space vectors contain the canonical variables:
    - x: Position in x direction in meters.
    - px: Horizontal momentum normalized over the reference momentum (dimensionless).
        :math:`px = \frac{P_x}{P_0}`
    - y: Position in y direction in meters.
    - py: Vertical momentum normalized over the reference momentum (dimensionless).
        :math:`py = \frac{P_y}{P_0}`
    - tau: Position in longitudinal direction in meters, relative to the reference
        particle. :math:`\tau = ct - \frac{s}{\beta_0}`, where s is the position along
        the beamline. In this notation, particle ahead of the reference particle will
        have negative :math:`\tau`.
    - p: Relative energy deviation from the reference particle (dimensionless).
        :math:`p = \frac{\Delta E}{p_0 C}`, where :math:`p_0` is the reference momentum.
        :math:`\Delta E = E - E_0`
    """

    @classmethod
    @abstractmethod
    def from_parameters(
        cls,
        mu_x: Optional[jnp.Array] = None,
        mu_px: Optional[jnp.Array] = None,
        mu_y: Optional[jnp.Array] = None,
        mu_py: Optional[jnp.Array] = None,
        mu_tau: Optional[jnp.Array] = None,
        mu_p: Optional[jnp.Array] = None,
        sigma_x: Optional[jnp.Array] = None,
        sigma_px: Optional[jnp.Array] = None,
        sigma_y: Optional[jnp.Array] = None,
        sigma_py: Optional[jnp.Array] = None,
        sigma_tau: Optional[jnp.Array] = None,
        sigma_p: Optional[jnp.Array] = None,
        cov_xpx: Optional[jnp.Array] = None,
        cov_ypy: Optional[jnp.Array] = None,
        cov_taup: Optional[jnp.Array] = None,
        energy: Optional[jnp.Array] = None,
        total_charge: Optional[jnp.Array] = None,
        device=None,
        dtype=None,
    ) -> "Beam":
        """
        Create beam that with given beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
        :param mu_tau: Center of the particle distribution on tau in meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in yp direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
            dimensionless.
        :param cov_xpx: Covariance between x and px.
        :param cov_ypy: Covariance between y and yp.
        :param cov_taup: Covariance between tau and p.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to create the beam on. If set to `"auto"` a CUDA GPU is
            selected if available. The CPU is used otherwise.
        :param dtype: Data type of the beam.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_twiss(
        cls,
        beta_x: Optional[jnp.Array] = None,
        alpha_x: Optional[jnp.Array] = None,
        emittance_x: Optional[jnp.Array] = None,
        beta_y: Optional[jnp.Array] = None,
        alpha_y: Optional[jnp.Array] = None,
        emittance_y: Optional[jnp.Array] = None,
        sigma_tau: Optional[jnp.Array] = None,
        sigma_p: Optional[jnp.Array] = None,
        cov_taup: Optional[jnp.Array] = None,
        energy: Optional[jnp.Array] = None,
        total_charge: Optional[jnp.Array] = None,
        device=None,
        dtype=None,
    ) -> "Beam":
        """
        Create a beam from twiss parameters.

        :param beta_x: Beta function in x direction in meters.
        :param alpha_x: Alpha function in x direction in rad.
        :param emittance_x: Emittance in x direction in m*rad.
        :param beta_y: Beta function in y direction in meters.
        :param alpha_y: Alpha function in y direction in rad.
        :param emittance_y: Emittance in y direction in m*rad.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
            dimensionless.
        :param cov_taup: Covariance between tau and p.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to create the beam on. If set to `"auto"` a CUDA GPU is
            selected if available. The CPU is used otherwise.
        :param dtype: Data type of the beam.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_ocelot(cls, parray, device=None, dtype=None) -> "Beam":
        """
        Convert an Ocelot ParticleArray `parray` to a Lynx Beam.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_astra(cls, path: str, device=None, dtype=None) -> "Beam":
        """Load an Astra particle distribution as a Lynx Beam."""
        raise NotImplementedError

    def transformed_to(
        self,
        mu_x: Optional[jnp.Array] = None,
        mu_px: Optional[jnp.Array] = None,
        mu_y: Optional[jnp.Array] = None,
        mu_py: Optional[jnp.Array] = None,
        mu_tau: Optional[jnp.Array] = None,
        mu_p: Optional[jnp.Array] = None,
        sigma_x: Optional[jnp.Array] = None,
        sigma_px: Optional[jnp.Array] = None,
        sigma_y: Optional[jnp.Array] = None,
        sigma_py: Optional[jnp.Array] = None,
        sigma_tau: Optional[jnp.Array] = None,
        sigma_p: Optional[jnp.Array] = None,
        energy: Optional[jnp.Array] = None,
        total_charge: Optional[jnp.Array] = None,
        device=None,
        dtype=None,
    ) -> "Beam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
        :param mu_tau: Center of the particle distribution on tau in meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in yp direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
            dimensionless.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to create the transformed beam on. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the transformed beam.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        # Figure out vector dimensions of the original beam and check that passed
        # arguments have the same vector dimensions.
        shape = self.mu_x.shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_px,
                mu_y,
                mu_py,
                mu_tau,
                mu_p,
                sigma_x,
                sigma_px,
                sigma_y,
                sigma_py,
                sigma_tau,
                sigma_p,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        if len(not_nones) > 0:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_px = mu_px if mu_px is not None else self.mu_px
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_py = mu_py if mu_py is not None else self.mu_py
        mu_tau = mu_tau if mu_tau is not None else self.mu_tau
        mu_p = mu_p if mu_p is not None else self.mu_p
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_px = sigma_px if sigma_px is not None else self.sigma_px
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_py = sigma_py if sigma_py is not None else self.sigma_py
        sigma_tau = sigma_tau if sigma_tau is not None else self.sigma_tau
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        total_charge = total_charge if total_charge is not None else self.total_charge

        return self.__class__.from_parameters(
            mu_x=mu_x,
            mu_px=mu_px,
            mu_y=mu_y,
            mu_py=mu_py,
            mu_tau=mu_tau,
            mu_p=mu_p,
            sigma_x=sigma_x,
            sigma_px=sigma_px,
            sigma_y=sigma_y,
            sigma_py=sigma_py,
            sigma_tau=sigma_tau,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    @property
    @abstractmethod
    def mu_x(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_x(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_px(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_px(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_y(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_y(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_py(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_py(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_tau(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_tau(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_p(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_p(self) -> jnp.Array:
        raise NotImplementedError

    @property
    def relativistic_gamma(self) -> jnp.Array:
        """Reference relativistic gamma of the beam."""
        return self.energy / electron_mass_eV

    @property
    def relativistic_beta(self) -> jnp.Array:
        """Reference relativistic beta of the beam."""
        relativistic_beta = jnp.ones_like(self.relativistic_gamma)
        relativistic_beta[jnp.abs(self.relativistic_gamma) > 0] = jnp.sqrt(
            1 - 1 / (self.relativistic_gamma[self.relativistic_gamma > 0] ** 2)
        )
        return relativistic_beta

    @property
    def p0c(self) -> jnp.Array:
        """Get the reference momentum * speed of light in eV."""
        return self.relativistic_beta * self.relativistic_gamma * electron_mass_eV

    @property
    @abstractmethod
    def cov_xpx(self) -> jnp.Array:
        # The covariance of (x,px) ~ $\sigma_{xpx}$
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_ypy(self) -> jnp.Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_taup(self) -> jnp.Array:
        raise NotImplementedError

    @property
    def emittance_x(self) -> jnp.Array:
        """Emittance of the beam in x direction in m."""
        return jnp.sqrt(
            jnp.clamp_min(
                self.sigma_x**2 * self.sigma_px**2 - self.cov_xpx**2,
                jnp.finfo(self.sigma_x.dtype).tiny,
            )
        )

    @property
    def normalized_emittance_x(self) -> jnp.Array:
        """Normalized emittance of the beam in x direction in m."""
        return self.emittance_x * self.relativistic_beta * self.relativistic_gamma

    @property
    def beta_x(self) -> jax.Array:
        """Beta function in x direction in meters."""
        return self.sigma_x**2 / self.emittance_x

    @property
    def alpha_x(self) -> jnp.Array:
        """Alpha function in x direction, dimensionless."""
        return -self.cov_xpx / self.emittance_x

    @property
    def emittance_y(self) -> jnp.Array:
        """Emittance of the beam in y direction in m."""
        return jnp.sqrt(
            jnp.clamp_min(
                self.sigma_y**2 * self.sigma_py**2 - self.cov_ypy**2,
                jnp.finfo(self.sigma_y.dtype).tiny,
            )
        )

    @property
    def normalized_emittance_y(self) -> jnp.Array:
        """Normalized emittance of the beam in y direction in m."""
        return self.emittance_y * self.relativistic_beta * self.relativistic_gamma

    @property
    def beta_y(self) -> jax.Array:
        """Beta function in y direction in meters."""
        return self.sigma_y**2 / self.emittance_y

    @property
    def alpha_y(self) -> jnp.Array:
        """Alpha function in y direction, dimensionless."""
        return -self.cov_ypy / self.emittance_y

    @abstractmethod
    def clone(self) -> "Beam":
        """Return a cloned beam that does not share the underlying memory."""
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
