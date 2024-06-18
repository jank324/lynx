from typing import Optional

import jax
import jax.numpy as jnp
from scipy.constants import physical_constants
from torch.distributions import MultivariateNormal

from .beam import Beam

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class ParticleBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param particles: List of 7-dimensional particle vectors.
    :param energy: Energy of the beam in eV.
    :param total_charge: Total charge of the beam in C.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        particles: jax.Array,
        energy: jax.Array,
        particle_charges: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        assert (
            particles.shape[-2] > 0 and particles.shape[-1] == 7
        ), "Particle vectors must be 7-dimensional."

        self.particles = particles.to(**factory_kwargs)
        self.particle_charges = (
            particle_charges.to(**factory_kwargs)
            if particle_charges is not None
            else jnp.zeros(particles.shape[:2], **factory_kwargs)
        )
        self.energy = energy.to(**factory_kwargs)

    @classmethod
    def from_parameters(
        cls,
        num_particles: Optional[jax.Array] = None,
        mu_x: Optional[jax.Array] = None,
        mu_y: Optional[jax.Array] = None,
        mu_xp: Optional[jax.Array] = None,
        mu_yp: Optional[jax.Array] = None,
        sigma_x: Optional[jax.Array] = None,
        sigma_y: Optional[jax.Array] = None,
        sigma_xp: Optional[jax.Array] = None,
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
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of random particles.

        :param num_particles: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_xp: Center of the particle distribution on x' in rad.
        :param mu_yp: Center of the particle distribution on y' in metraders.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param cor_x: Correlation between x and x'.
        :param cor_y: Correlation between y and y'.
        :param cor_s: Correlation between s and p.
        :param energy: Energy of the beam in eV.
        :total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
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
        num_particles = num_particles if num_particles is not None else 100_000
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
        particle_charges = (
            jnp.ones((*shape, num_particles), device=device, dtype=dtype)
            * total_charge.unsqueeze(-1)
            / num_particles
        )

        mean = jnp.stack(
            [mu_x, mu_xp, mu_y, mu_yp, jnp.zeros(shape), jnp.zeros(shape)], dim=-1
        )

        cov = jnp.zeros(*shape, 6, 6)
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

        particles = jnp.ones((*shape, num_particles, 7))
        distributions = [
            MultivariateNormal(sample_mean, covariance_matrix=sample_cov)
            for sample_mean, sample_cov in zip(mean.view(-1, 6), cov.view(-1, 6, 6))
        ]
        particles[..., :6] = jnp.stack(
            [distribution.sample((num_particles,)) for distribution in distributions],
            dim=0,
        ).view(*shape, num_particles, 6)

        return cls(
            particles,
            energy,
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_twiss(
        cls,
        num_particles: Optional[jax.Array] = None,
        beta_x: Optional[jax.Array] = None,
        alpha_x: Optional[jax.Array] = None,
        emittance_x: Optional[jax.Array] = None,
        beta_y: Optional[jax.Array] = None,
        alpha_y: Optional[jax.Array] = None,
        emittance_y: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        sigma_s: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        cor_s: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> "ParticleBeam":
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
                energy,
                sigma_s,
                sigma_p,
                cor_s,
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
        num_particles = num_particles if num_particles is not None else 1_000_000
        beta_x = beta_x if beta_x is not None else jnp.full(shape, 0.0)
        alpha_x = alpha_x if alpha_x is not None else jnp.full(shape, 0.0)
        emittance_x = emittance_x if emittance_x is not None else jnp.full(shape, 0.0)
        beta_y = beta_y if beta_y is not None else jnp.full(shape, 0.0)
        alpha_y = alpha_y if alpha_y is not None else jnp.full(shape, 0.0)
        emittance_y = emittance_y if emittance_y is not None else jnp.full(shape, 0.0)
        energy = energy if energy is not None else jnp.full(shape, 1e8)
        sigma_s = sigma_s if sigma_s is not None else jnp.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else jnp.full(shape, 1e-6)
        cor_s = cor_s if cor_s is not None else jnp.full(shape, 0.0)
        total_charge = (
            total_charge if total_charge is not None else jnp.full(shape, 0.0)
        )

        sigma_x = jnp.sqrt(beta_x * emittance_x)
        sigma_xp = jnp.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = jnp.sqrt(beta_y * emittance_y)
        sigma_yp = jnp.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cor_x = -emittance_x * alpha_x
        cor_y = -emittance_y * alpha_y

        return cls.from_parameters(
            num_particles=num_particles,
            mu_x=jnp.full(shape, 0.0),
            mu_xp=jnp.full(shape, 0.0),
            mu_y=jnp.full(shape, 0.0),
            mu_yp=jnp.full(shape, 0.0),
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
            dtype=dtype,
        )

    @classmethod
    def uniform_3d_ellipsoid(
        cls,
        num_particles: Optional[jax.Array] = None,
        radius_x: Optional[jax.Array] = None,
        radius_y: Optional[jax.Array] = None,
        radius_s: Optional[jax.Array] = None,
        sigma_xp: Optional[jax.Array] = None,
        sigma_yp: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ):
        """
        Generate a particle beam with spatially uniformly distributed particles inside
        an ellipsoid, i.e. a waterbag distribution.

        Note that:
         - The generated particles do not have correlation in the momentum directions,
           and by default a cold beam with no divergence is generated.
         - For batched generation, parameters that are not `None` must have the same
           shape.

        :param num_particles: Number of particles to generate.
        :param radius_x: Radius of the ellipsoid in x direction in meters.
        :param radius_y: Radius of the ellipsoid in y direction in meters.
        :param radius_s: Radius of the ellipsoid in s (longitudinal) direction
        in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad,
        default is 0.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad,
        default is 0.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to.
        :param dtype: Data type of the generated particles.

        :return: ParticleBeam with uniformly distributed particles inside an ellipsoid.
        """

        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                radius_x,
                radius_y,
                radius_s,
                sigma_xp,
                sigma_yp,
                sigma_p,
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
        # NOTE that this does not need to be done for values that are passed to the
        # Gaussian beam generation.
        num_particles = num_particles if num_particles is not None else 1_000_000
        radius_x = radius_x if radius_x is not None else jnp.full(shape, 1e-3)
        radius_y = radius_y if radius_y is not None else jnp.full(shape, 1e-3)
        radius_s = radius_s if radius_s is not None else jnp.full(shape, 1e-3)

        # Generate xs, ys and ss within the ellipsoid
        flattened_xs = jnp.empty(*shape, num_particles).flatten(end_dim=-2)
        flattened_ys = jnp.empty(*shape, num_particles).flatten(end_dim=-2)
        flattened_ss = jnp.empty(*shape, num_particles).flatten(end_dim=-2)
        for i, (r_x, r_y, r_s) in enumerate(
            zip(radius_x.flatten(), radius_y.flatten(), radius_s.flatten())
        ):
            num_successful = 0
            while num_successful < num_particles:
                xs = (jnp.rand(num_particles) - 0.5) * 2 * r_x
                ys = (jnp.rand(num_particles) - 0.5) * 2 * r_y
                ss = (jnp.rand(num_particles) - 0.5) * 2 * r_s

                is_in_ellipsoid = xs**2 / r_x**2 + ys**2 / r_y**2 + ss**2 / r_s**2 < 1
                num_to_add = min(num_particles - num_successful, is_in_ellipsoid.sum())

                flattened_xs[i, num_successful : num_successful + num_to_add] = xs[
                    is_in_ellipsoid
                ][:num_to_add]
                flattened_ys[i, num_successful : num_successful + num_to_add] = ys[
                    is_in_ellipsoid
                ][:num_to_add]
                flattened_ss[i, num_successful : num_successful + num_to_add] = ss[
                    is_in_ellipsoid
                ][:num_to_add]

                num_successful += num_to_add

        # Generate an uncorrelated Gaussian beam
        beam = cls.from_parameters(
            num_particles=num_particles,
            mu_xp=jnp.full(shape, 0.0),
            mu_yp=jnp.full(shape, 0.0),
            sigma_xp=sigma_xp,
            sigma_yp=sigma_yp,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

        # Replace the spatial coordinates with the generated ones
        beam.xs = flattened_xs.view(*shape, num_particles)
        beam.ys = flattened_ys.view(*shape, num_particles)
        beam.ss = flattened_ss.view(*shape, num_particles)

        return beam

    @classmethod
    def make_linspaced(
        cls,
        num_particles: Optional[jax.Array] = None,
        mu_x: Optional[jax.Array] = None,
        mu_y: Optional[jax.Array] = None,
        mu_xp: Optional[jax.Array] = None,
        mu_yp: Optional[jax.Array] = None,
        sigma_x: Optional[jax.Array] = None,
        sigma_y: Optional[jax.Array] = None,
        sigma_xp: Optional[jax.Array] = None,
        sigma_yp: Optional[jax.Array] = None,
        sigma_s: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of *n* linspaced particles.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_xp: Center of the particle distribution on x' in rad.
        :param mu_yp: Center of the particle distribution on y' in rad.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Energy of the beam in eV.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
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
        num_particles = num_particles if num_particles is not None else 10
        mu_x = mu_x if mu_x is not None else jnp.full(shape, 0.0)
        mu_xp = mu_xp if mu_xp is not None else jnp.full(shape, 0.0)
        mu_y = mu_y if mu_y is not None else jnp.full(shape, 0.0)
        mu_yp = mu_yp if mu_yp is not None else jnp.full(shape, 0.0)
        sigma_x = sigma_x if sigma_x is not None else jnp.full(shape, 175e-9)
        sigma_xp = sigma_xp if sigma_xp is not None else jnp.full(shape, 2e-7)
        sigma_y = sigma_y if sigma_y is not None else jnp.full(shape, 175e-9)
        sigma_yp = sigma_yp if sigma_yp is not None else jnp.full(shape, 2e-7)
        sigma_s = sigma_s if sigma_s is not None else jnp.full(shape, 0.0)
        sigma_p = sigma_p if sigma_p is not None else jnp.full(shape, 0.0)
        energy = energy if energy is not None else jnp.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else jnp.full(shape, 0.0)
        )

        particle_charges = (
            jnp.ones((shape[0], num_particles), device=device, dtype=dtype)
            * total_charge.view(-1, 1)
            / num_particles
        )

        particles = jnp.ones((shape[0], num_particles, 7))

        particles[:, :, 0] = jnp.stack(
            [
                jnp.linspace(
                    sample_mu_x - sample_sigma_x,
                    sample_mu_x + sample_sigma_x,
                    num_particles,
                )
                for sample_mu_x, sample_sigma_x in zip(mu_x, sigma_x)
            ],
            dim=0,
        )
        particles[:, :, 1] = jnp.stack(
            [
                jnp.linspace(
                    sample_mu_xp - sample_sigma_xp,
                    sample_mu_xp + sample_sigma_xp,
                    num_particles,
                )
                for sample_mu_xp, sample_sigma_xp in zip(mu_xp, sigma_xp)
            ],
            dim=0,
        )
        particles[:, :, 2] = jnp.stack(
            [
                jnp.linspace(
                    sample_mu_y - sample_sigma_y,
                    sample_mu_y + sample_sigma_y,
                    num_particles,
                )
                for sample_mu_y, sample_sigma_y in zip(mu_y, sigma_y)
            ],
            dim=0,
        )
        particles[:, :, 3] = jnp.stack(
            [
                jnp.linspace(
                    sample_mu_yp - sample_sigma_yp,
                    sample_mu_yp + sample_sigma_yp,
                    num_particles,
                )
                for sample_mu_yp, sample_sigma_yp in zip(mu_yp, sigma_yp)
            ],
            dim=0,
        )
        particles[:, :, 4] = jnp.stack(
            [
                jnp.linspace(
                    -sample_sigma_s, sample_sigma_s, num_particles, device=device
                )
                for sample_sigma_s in sigma_s
            ],
            dim=0,
        )
        particles[:, :, 5] = jnp.stack(
            [
                jnp.linspace(
                    -sample_sigma_p, sample_sigma_p, num_particles, device=device
                )
                for sample_sigma_p in sigma_p
            ],
            dim=0,
        )

        return cls(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_ocelot(cls, parray, device=None, dtype=jnp.float32) -> "ParticleBeam":
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        num_particles = parray.rparticles.shape[1]
        particles = jnp.ones((num_particles, 7))
        particles[:, :6] = jnp.array(parray.rparticles.transpose())
        particle_charges = jnp.array(parray.q_array)

        return cls(
            particles=particles.unsqueeze(0),
            energy=jnp.array(1e9 * parray.E).unsqueeze(0),
            particle_charges=particle_charges.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_astra(cls, path: str, device=None, dtype=jnp.float32) -> "ParticleBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from lynx.converters.astra import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)
        particles_7d = jnp.ones((particles.shape[0], 7))
        particles_7d[:, :6] = jnp.array(particles)
        particle_charges = jnp.array(particle_charges)
        return cls(
            particles=particles_7d.unsqueeze(0),
            energy=jnp.array(energy).unsqueeze(0),
            particle_charges=particle_charges.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    def transformed_to(
        self,
        mu_x: Optional[jax.Array] = None,
        mu_y: Optional[jax.Array] = None,
        mu_xp: Optional[jax.Array] = None,
        mu_yp: Optional[jax.Array] = None,
        sigma_x: Optional[jax.Array] = None,
        sigma_y: Optional[jax.Array] = None,
        sigma_xp: Optional[jax.Array] = None,
        sigma_yp: Optional[jax.Array] = None,
        sigma_s: Optional[jax.Array] = None,
        sigma_p: Optional[jax.Array] = None,
        energy: Optional[jax.Array] = None,
        total_charge: Optional[jax.Array] = None,
        device=None,
        dtype=jnp.float32,
    ) -> "ParticleBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_xp: Center of the particle distribution on x' in rad.
        :param mu_yp: Center of the particle distribution on y' in rad.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
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
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_xp = mu_xp if mu_xp is not None else self.mu_xp
        mu_yp = mu_yp if mu_yp is not None else self.mu_yp
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_xp = sigma_xp if sigma_xp is not None else self.sigma_xp
        sigma_yp = sigma_yp if sigma_yp is not None else self.sigma_yp
        sigma_s = sigma_s if sigma_s is not None else self.sigma_s
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        if total_charge is None:
            particle_charges = self.particle_charges
        elif self.total_charge is None:  # Scale to the new charge
            total_charge = total_charge.to(
                device=self.particle_charges.device, dtype=self.particle_charges.dtype
            )
            particle_charges = self.particle_charges * total_charge / self.total_charge
        else:
            particle_charges = (
                jnp.ones_like(self.particle_charges, device=device, dtype=dtype)
                * total_charge.view(-1, 1)
                / self.particle_charges.shape[-1]
            )

        new_mu = jnp.stack(
            [mu_x, mu_xp, mu_y, mu_yp, jnp.full(shape, 0.0), jnp.full(shape, 0.0)],
            dim=1,
        )
        new_sigma = jnp.stack(
            [sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s, sigma_p], dim=1
        )

        old_mu = jnp.stack(
            [
                self.mu_x,
                self.mu_xp,
                self.mu_y,
                self.mu_yp,
                jnp.full(shape, 0.0),
                jnp.full(shape, 0.0),
            ],
            dim=1,
        )
        old_sigma = jnp.stack(
            [
                self.sigma_x,
                self.sigma_xp,
                self.sigma_y,
                self.sigma_yp,
                self.sigma_s,
                self.sigma_p,
            ],
            dim=1,
        )

        phase_space = self.particles[:, :, :6]
        phase_space = (phase_space - old_mu.unsqueeze(1)) / old_sigma.unsqueeze(
            1
        ) * new_sigma.unsqueeze(1) + new_mu.unsqueeze(1)

        particles = jnp.ones_like(self.particles)
        particles[:, :, :6] = phase_space

        return self.__class__(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    def __len__(self) -> int:
        return int(self.num_particles)

    @property
    def total_charge(self) -> jax.Array:
        return jnp.sum(self.particle_charges, dim=-1)

    @property
    def num_particles(self) -> int:
        return self.particles.shape[-2]

    @property
    def xs(self) -> Optional[jax.Array]:
        return self.particles[..., 0] if self is not Beam.empty else None

    @xs.setter
    def xs(self, value: jax.Array) -> None:
        self.particles[..., 0] = value

    @property
    def mu_x(self) -> Optional[jax.Array]:
        return self.xs.mean(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_x(self) -> Optional[jax.Array]:
        return self.xs.std(dim=-1) if self is not Beam.empty else None

    @property
    def xps(self) -> Optional[jax.Array]:
        return self.particles[..., 1] if self is not Beam.empty else None

    @xps.setter
    def xps(self, value: jax.Array) -> None:
        self.particles[..., 1] = value

    @property
    def mu_xp(self) -> Optional[jax.Array]:
        return self.xps.mean(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_xp(self) -> Optional[jax.Array]:
        return self.xps.std(dim=-1) if self is not Beam.empty else None

    @property
    def ys(self) -> Optional[jax.Array]:
        return self.particles[..., 2] if self is not Beam.empty else None

    @ys.setter
    def ys(self, value: jax.Array) -> None:
        self.particles[..., 2] = value

    @property
    def mu_y(self) -> Optional[float]:
        return self.ys.mean(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_y(self) -> Optional[jax.Array]:
        return self.ys.std(dim=-1) if self is not Beam.empty else None

    @property
    def yps(self) -> Optional[jax.Array]:
        return self.particles[..., 3] if self is not Beam.empty else None

    @yps.setter
    def yps(self, value: jax.Array) -> None:
        self.particles[..., 3] = value

    @property
    def mu_yp(self) -> Optional[jax.Array]:
        return self.yps.mean(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_yp(self) -> Optional[jax.Array]:
        return self.yps.std(dim=-1) if self is not Beam.empty else None

    @property
    def ss(self) -> Optional[jax.Array]:
        return self.particles[..., 4] if self is not Beam.empty else None

    @ss.setter
    def ss(self, value: jax.Array) -> None:
        self.particles[..., 4] = value

    @property
    def mu_s(self) -> Optional[jax.Array]:
        return self.ss.mean(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_s(self) -> Optional[jax.Array]:
        return self.ss.std(dim=-1) if self is not Beam.empty else None

    @property
    def ps(self) -> Optional[jax.Array]:
        return self.particles[..., 5] if self is not Beam.empty else None

    @ps.setter
    def ps(self, value: jax.Array) -> None:
        self.particles[..., 5] = value

    @property
    def mu_p(self) -> Optional[jax.Array]:
        return self.ps.mean(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_p(self) -> Optional[jax.Array]:
        return self.ps.std(dim=-1) if self is not Beam.empty else None

    @property
    def sigma_xxp(self) -> jax.Array:
        return jnp.mean(
            (self.xs - self.mu_x.view(-1, 1)) * (self.xps - self.mu_xp.view(-1, 1)),
            dim=1,
        )

    @property
    def sigma_yyp(self) -> jax.Array:
        return jnp.mean(
            (self.ys - self.mu_y.view(-1, 1)) * (self.yps - self.mu_yp.view(-1, 1)),
            dim=1,
        )

    def broadcast(self, shape: tuple) -> "ParticleBeam":
        return self.__class__(
            particles=self.particles.repeat((*shape, 1, 1)),
            energy=self.energy.repeat(shape),
            particle_charges=self.particle_charges.repeat((*shape, 1)),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={repr(self.num_particles)},"
            f" mu_x={repr(self.mu_x)}, mu_xp={repr(self.mu_xp)},"
            f" mu_y={repr(self.mu_y)}, mu_yp={repr(self.mu_yp)},"
            f" sigma_x={repr(self.sigma_x)}, sigma_xp={repr(self.sigma_xp)},"
            f" sigma_y={repr(self.sigma_y)}, sigma_yp={repr(self.sigma_yp)},"
            f" sigma_s={repr(self.sigma_s)}, sigma_p={repr(self.sigma_p)},"
            f" energy={repr(self.energy)})"
            f" total_charge={repr(self.total_charge)})"
        )
