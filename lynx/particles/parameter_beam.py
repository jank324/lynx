from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants

from .beam import Beam

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class ParameterBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param mu: Mu vector of the beam.
    :param cov: Covariance matrix of the beam.
    :param energy: Energy of the beam in eV.
    :param total_charge: Total charge of the beam in C.
    :param device: Device to use for the beam. If "auto", use CUDA if available.
        Note: Compuationally it would be faster to use CPU for ParameterBeam.
    """

    def __init__(
        self,
        mu: jax.Array,
        cov: jax.Array,
        energy: jax.Array,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self._mu = jnp.asarray(mu, **factory_kwargs)
        self._cov = jnp.asarray(cov, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else jnp.asarray([0.0], **factory_kwargs)
        )
        self.total_charge = jnp.asarray(total_charge, **factory_kwargs)
        self.energy = jnp.asarray(energy, **factory_kwargs)

    @classmethod
    def from_parameters(
        cls,
        mu_x: Optional[jax.Array] = None,
        mu_xp: Optional[jax.Array] = None,
        mu_y: Optional[jax.Array] = None,
        mu_yp: Optional[jax.Array] = None,
        sigma_x: Optional[jax.Array] = None,
        sigma_xp: Optional[jax.Array] = None,
        sigma_y: Optional[jax.Array] = None,
        sigma_yp: Optional[jax.Array] = None,
        sigma_s: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        cor_x: Optional[jax.Array] = None,
        cor_y: Optional[jax.Array] = None,
        cor_s: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> "ParameterBeam":
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                cor_x,
                cor_y,
                cor_s,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else (1,)
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        mu_x = mu_x if mu_x is not None else jnp.full(shape, 0.0)
        mu_xp = mu_xp if mu_xp is not None else jnp.full(shape, 0.0)
        mu_y = mu_y if mu_y is not None else jnp.full(shape, 0.0)
        mu_yp = mu_yp if mu_yp is not None else jnp.full(shape, 0.0)
        sigma_x = sigma_x if sigma_x is not None else jnp.full(shape, 175e-9)
        sigma_xp = sigma_xp if sigma_xp is not None else jnp.full(shape, 2e-7)
        sigma_y = sigma_y if sigma_y is not None else jnp.full(shape, 175e-9)
        sigma_yp = sigma_yp if sigma_yp is not None else jnp.full(shape, 2e-7)
        sigma_s = sigma_s if sigma_s is not None else jnp.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else jnp.full(shape, 1e-6)
        cor_x = cor_x if cor_x is not None else jnp.full(shape, 0.0)
        cor_y = cor_y if cor_y is not None else jnp.full(shape, 0.0)
        cor_s = cor_s if cor_s is not None else jnp.full(shape, 0.0)
        energy = energy if energy is not None else jnp.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else jnp.full(shape, 0.0)
        )

        mu = jnp.stack(
            [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                jnp.full(shape, 0.0),
                jnp.full(shape, 0.0),
                jnp.full(shape, 1.0),
            ],
            dim=-1,
        )

        cov = jnp.zeros(*shape, 7, 7)
        cov[..., 0, 0] = sigma_x**2
        cov[..., 0, 1] = cor_x
        cov[..., 1, 0] = cor_x
        cov[..., 1, 1] = sigma_xp**2
        cov[..., 2, 2] = sigma_y**2
        cov[..., 2, 3] = cor_y
        cov[..., 3, 2] = cor_y
        cov[..., 3, 3] = sigma_yp**2
        cov[..., 4, 4] = sigma_s**2
        cov[..., 4, 5] = cor_s
        cov[..., 5, 4] = cor_s
        cov[..., 5, 5] = sigma_p**2

        return cls(
            mu=mu, cov=cov, energy=energy, total_charge=total_charge, device=device
        )

    @classmethod
    def from_twiss(
        cls,
        beta_x: Optional[jax.Array] = None,
        alpha_x: Optional[jax.Array] = None,
        emittance_x: Optional[jax.Array] = None,
        beta_y: Optional[jax.Array] = None,
        alpha_y: Optional[jax.Array] = None,
        emittance_y: Optional[jax.Array] = None,
        sigma_s: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        cor_s: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> "ParameterBeam":
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                beta_x,
                alpha_x,
                emittance_x,
                beta_y,
                alpha_y,
                emittance_y,
                sigma_s,
                sigma_p,
                cor_s,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else (1,)
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        beta_x = beta_x if beta_x is not None else jnp.full(shape, 1.0)
        alpha_x = alpha_x if alpha_x is not None else jnp.full(shape, 0.0)
        emittance_x = (
            emittance_x if emittance_x is not None else jnp.full(shape, 7.1971891e-13)
        )
        beta_y = beta_y if beta_y is not None else jnp.full(shape, 1.0)
        alpha_y = alpha_y if alpha_y is not None else jnp.full(shape, 0.0)
        emittance_y = (
            emittance_y if emittance_y is not None else jnp.full(shape, 7.1971891e-13)
        )
        sigma_s = sigma_s if sigma_s is not None else jnp.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else jnp.full(shape, 1e-6)
        cor_s = cor_s if cor_s is not None else jnp.full(shape, 0.0)
        energy = energy if energy is not None else jnp.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else jnp.full(shape, 0.0)
        )

        assert all(
            beta_x > 0
        ), "Beta function in x direction must be larger than 0 everywhere."
        assert all(
            beta_y > 0
        ), "Beta function in y direction must be larger than 0 everywhere."

        sigma_x = jnp.sqrt(emittance_x * beta_x)
        sigma_xp = jnp.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = jnp.sqrt(emittance_y * beta_y)
        sigma_yp = jnp.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cor_x = -emittance_x * alpha_x
        cor_y = -emittance_y * alpha_y
        return cls.from_parameters(
            sigma_x=sigma_x,
            sigma_xp=sigma_xp,
            sigma_y=sigma_y,
            sigma_yp=sigma_yp,
            sigma_s=sigma_s,
            sigma_p=sigma_p,
            energy=energy,
            cor_s=cor_s,
            cor_x=cor_x,
            cor_y=cor_y,
            total_charge=total_charge,
            device=device,
        )

    @classmethod
    def from_ocelot(cls, parray, device=None, dtype=jnp.float32) -> "ParameterBeam":
        """Load an Ocelot ParticleArray `parray` as a Cheetah Beam."""
        mu = jnp.ones(7)
        mu[:6] = jnp.array(parray.rparticles.mean(axis=1), dtype=jnp.float32)

        cov = jnp.zeros(7, 7)
        cov[:6, :6] = jnp.array(np.cov(parray.rparticles), dtype=jnp.float32)

        energy = jnp.array(1e9 * parray.E, dtype=jnp.float32)
        total_charge = jnp.array(np.sum(parray.q_array), dtype=jnp.float32)

        return cls(
            mu=mu.unsqueeze(0),
            cov=cov.unsqueeze(0),
            energy=energy.unsqueeze(0),
            total_charge=total_charge.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_astra(cls, path: str, device=None, dtype=jnp.float32) -> "ParameterBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from lynx.converters.astra import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)
        mu = jnp.ones(7)
        mu[:6] = jnp.array(particles.mean(axis=0))

        cov = jnp.zeros(7, 7)
        cov[:6, :6] = jnp.array(jnp.cov(particles.transpose()), dtype=jnp.float32)

        total_charge = jnp.array(np.sum(particle_charges), dtype=jnp.float32)

        return cls(
            mu=mu.unsqueeze(0),
            cov=cov.unsqueeze(0),
            energy=jnp.array(energy, dtype=jnp.float32).unsqueeze(0),
            total_charge=total_charge.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    def transformed_to(
        self,
        mu_x: Optional[jax.Array] = None,
        mu_xp: Optional[jax.Array] = None,
        mu_y: Optional[jax.Array] = None,
        mu_yp: Optional[jax.Array] = None,
        sigma_x: Optional[jax.Array] = None,
        sigma_xp: Optional[jax.Array] = None,
        sigma_y: Optional[jax.Array] = None,
        sigma_yp: Optional[jax.Array] = None,
        sigma_s: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> "ParameterBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_xp: Center of the particle distribution on x' in rad.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_yp: Center of the particle distribution on y' in rad.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        # Figure out batch size of the original beam and check that passed arguments
        # have the same batch size
        shape = self.mu_x.shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
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
        mu_xp = mu_xp if mu_xp is not None else self.mu_xp
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_yp = mu_yp if mu_yp is not None else self.mu_yp
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_xp = sigma_xp if sigma_xp is not None else self.sigma_xp
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_yp = sigma_yp if sigma_yp is not None else self.sigma_yp
        sigma_s = sigma_s if sigma_s is not None else self.sigma_s
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        total_charge = total_charge if total_charge is not None else self.total_charge

        return self.__class__.from_parameters(
            mu_x=mu_x,
            mu_xp=mu_xp,
            mu_y=mu_y,
            mu_yp=mu_yp,
            sigma_x=sigma_x,
            sigma_xp=sigma_xp,
            sigma_y=sigma_y,
            sigma_yp=sigma_yp,
            sigma_s=sigma_s,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    @property
    def mu_x(self) -> jax.Array:
        return self._mu[..., 0]

    @property
    def sigma_x(self) -> jax.Array:
        return jnp.sqrt(jnp.clamp_min(self._cov[..., 0, 0], 1e-20))

    @property
    def mu_xp(self) -> jax.Array:
        return self._mu[..., 1]

    @property
    def sigma_xp(self) -> jax.Array:
        return jnp.sqrt(jnp.clamp_min(self._cov[..., 1, 1], 1e-20))

    @property
    def mu_y(self) -> jax.Array:
        return self._mu[..., 2]

    @property
    def sigma_y(self) -> jax.Array:
        return jnp.sqrt(jnp.clamp_min(self._cov[..., 2, 2], 1e-20))

    @property
    def mu_yp(self) -> jax.Array:
        return self._mu[..., 3]

    @property
    def sigma_yp(self) -> jax.Array:
        return jnp.sqrt(jnp.clamp_min(self._cov[..., 3, 3], 1e-20))

    @property
    def mu_s(self) -> jax.Array:
        return self._mu[..., 4]

    @property
    def sigma_s(self) -> jax.Array:
        return jnp.sqrt(jnp.clamp_min(self._cov[..., 4, 4], 1e-20))

    @property
    def mu_p(self) -> jax.Array:
        return self._mu[..., 5]

    @property
    def sigma_p(self) -> jax.Array:
        return jnp.sqrt(jnp.clamp_min(self._cov[..., 5, 5], 1e-20))

    @property
    def sigma_xxp(self) -> jax.Array:
        return self._cov[..., 0, 1]

    @property
    def sigma_yyp(self) -> jax.Array:
        return self._cov[..., 2, 3]

    def broadcast(self, shape: tuple) -> "ParameterBeam":
        return self.__class__(
            mu=self._mu.repeat((*shape, 1)),
            cov=self._cov.repeat((*shape, 1, 1)),
            energy=self.energy.repeat(shape),
            total_charge=self.total_charge.repeat(shape),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={repr(self.mu_x)},"
            f" mu_xp={repr(self.mu_xp)}, mu_y={repr(self.mu_y)},"
            f" mu_yp={repr(self.mu_yp)}, sigma_x={repr(self.sigma_x)},"
            f" sigma_xp={repr(self.sigma_xp)}, sigma_y={repr(self.sigma_y)},"
            f" sigma_yp={repr(self.sigma_yp)}, sigma_s={repr(self.sigma_s)},"
            f" sigma_p={repr(self.sigma_p)}, energy={repr(self.energy)}),"
            f" total_charge={repr(self.total_charge)})"
        )