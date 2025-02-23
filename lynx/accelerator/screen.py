from typing import Literal, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jnp.distributions import MultivariateNormal
from matplotlib.patches import Rectangle

from lynx.accelerator.element import Element
from lynx.particles import Beam, ParameterBeam, ParticleBeam
from lynx.utils import UniqueNameGenerator, kde_histogram_2d, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Screen(Element):
    """
    Diagnostic screen in a particle accelerator.

    :param resolution: Resolution of the camera sensor looking at the screen given as
        Tensor `(width, height)` in pixels.
    :param pixel_size: Size of a pixel on the screen in meters given as a Tensor
        `(width, height)`.
    :param binning: Binning used by the camera.
    :param misalignment: Misalignment of the screen in meters given as a Tensor
        `(x, y)`.
    :param method: Method used to generate the screen's reading. Can be either
        "histogram" or "kde", defaults to "histogram". KDE will be slower but allows
        backward differentiation.
    :param kde_bandwidth: Bandwidth used for the kernel density estimation in meters.
        Controls the smoothness of the distribution.
    :param is_blocking: If `True` the screen is blocking and will stop the beam.
    :param is_active: If `True` the screen is active and will record the beam's
        distribution. If `False` the screen is inactive and will not record the beam's
        distribution.
    :param name: Unique identifier of the element.

    NOTE: `method='histogram'` currently does not support vectorisation. Please use
        `method=`kde` instead. Similarly, `ParameterBeam` can also not be vectorised.
        Please use `ParticleBeam` instead.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1024, 1024),
        pixel_size: Optional[jnp.Array] = None,
        binning: int = 1,
        misalignment: Optional[jnp.Array] = None,
        method: Literal["histogram", "kde"] = "histogram",
        kde_bandwidth: Optional[jnp.Array] = None,
        is_blocking: bool = False,
        is_active: bool = False,
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [pixel_size, misalignment, kde_bandwidth], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        assert method in [
            "histogram",
            "kde",
        ], f"Invalid method {method}. Must be either 'histogram' or 'kde'."

        self.resolution = resolution
        self.binning = binning
        self.method = method
        self.is_blocking = is_blocking
        self.is_active = is_active

        self.register_buffer("pixel_size", jnp.asarray((1e-3, 1e-3), **factory_kwargs))
        self.register_buffer("misalignment", jnp.asarray((0.0, 0.0), **factory_kwargs))
        self.register_buffer("kde_bandwidth", jnp.clone(self.pixel_size[0]))

        # NOTE: According to its type hint, the operation on resolution below is a
        # no-op. However, this form is robust against accidentally passing a
        # jnp.Array, preventing crashes in some instances.
        self.register_buffer(
            "cached_reading",
            jnp.full((resolution[0], resolution[1]), jnp.nan, **factory_kwargs),
        )

        if pixel_size is not None:
            self.pixel_size = jnp.as_tensor(pixel_size, **factory_kwargs)
        if misalignment is not None:
            self.misalignment = jnp.as_tensor(misalignment, **factory_kwargs)
        if kde_bandwidth is not None:
            self.kde_bandwidth = jnp.as_tensor(kde_bandwidth, **factory_kwargs)

        self.set_read_beam(None)

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    @property
    def effective_resolution(self) -> tuple[int, int]:
        return (
            self.resolution[0] // self.binning,
            self.resolution[1] // self.binning,
        )

    @property
    def effective_pixel_size(self) -> jax.Array:
        return self.pixel_size * self.binning

    @property
    def extent(self) -> jax.Array:
        return jnp.stack(
            [
                -self.resolution[0] * self.pixel_size[0] / 2,
                self.resolution[0] * self.pixel_size[0] / 2,
                -self.resolution[1] * self.pixel_size[1] / 2,
                self.resolution[1] * self.pixel_size[1] / 2,
            ]
        )

    @property
    def pixel_bin_edges(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.linspace(
                -self.resolution[0] * self.pixel_size[0] / 2,
                self.resolution[0] * self.pixel_size[0] / 2,
                int(self.effective_resolution[0]) + 1,
            ),
            jnp.linspace(
                -self.resolution[1] * self.pixel_size[1] / 2,
                self.resolution[1] * self.pixel_size[1] / 2,
                int(self.effective_resolution[1]) + 1,
            ),
        )

    @property
    def pixel_bin_centers(self) -> tuple[jnp.Array, jnp.Array]:
        return (
            (self.pixel_bin_edges[0][1:] + self.pixel_bin_edges[0][:-1]) / 2,
            (self.pixel_bin_edges[1][1:] + self.pixel_bin_edges[1][:-1]) / 2,
        )

    def transfer_map(self, energy: jnp.Array) -> jnp.Array:
        device = self.misalignment.device
        dtype = self.misalignment.dtype

        return jnp.eye(7, device=device, dtype=dtype).repeat((*energy.shape, 1, 1))

    def track(self, incoming: Beam) -> Beam:
        # Record the beam only when the screen is active
        if self.is_active:
            copy_of_incoming = incoming.clone()

            if isinstance(incoming, ParameterBeam):
                copy_of_incoming._mu, _ = jnp.broadcast_tensors(
                    copy_of_incoming._mu, self.misalignment[..., 0]
                )
                copy_of_incoming._mu = copy_of_incoming._mu.clone()

                copy_of_incoming._mu[..., 0] -= self.misalignment[..., 0]
                copy_of_incoming._mu[..., 2] -= self.misalignment[..., 1]
            elif isinstance(incoming, ParticleBeam):
                copy_of_incoming.particles, _ = jnp.broadcast_tensors(
                    copy_of_incoming.particles,
                    self.misalignment[..., 0].unsqueeze(-1).unsqueeze(-1),
                )
                copy_of_incoming.particles = copy_of_incoming.particles.clone()

                copy_of_incoming.particles[..., 0] -= self.misalignment[
                    ..., 0
                ].unsqueeze(-1)
                copy_of_incoming.particles[..., 1] -= self.misalignment[
                    ..., 1
                ].unsqueeze(-1)

            self.set_read_beam(copy_of_incoming)

        # Block the beam only when the screen is active and blocking
        if self.is_active and self.is_blocking:
            if isinstance(incoming, ParameterBeam):
                return ParameterBeam(
                    mu=incoming._mu,
                    cov=incoming._cov,
                    energy=incoming.energy,
                    total_charge=jnp.zeros_like(incoming.total_charge),
                )
            elif isinstance(incoming, ParticleBeam):
                return ParticleBeam(
                    particles=incoming.particles,
                    energy=incoming.energy,
                    particle_charges=incoming.particle_charges,
                    survival_probabilities=jnp.zeros_like(
                        incoming.survival_probabilities
                    ),
                )
        else:
            return incoming.clone()

    @property
    def reading(self) -> jnp.Array:
        if not jnp.all(jnp.isnan(self.cached_reading)):
            return self.cached_reading

        read_beam = self.get_read_beam()
        if read_beam is None:
            image = jnp.zeros(
                (int(self.effective_resolution[1]), int(self.effective_resolution[0])),
                device=self.misalignment.device,
                dtype=self.misalignment.dtype,
            )
        elif isinstance(read_beam, ParameterBeam):
            if jnp.numel(read_beam._mu[..., 0]) > 1:
                raise NotImplementedError(
                    "`Screen` does not support vectorization of `ParameterBeam`. "
                    "Please use `ParticleBeam` instead. If this is a feature you would "
                    "like to see, please open an issue on GitHub."
                )

            transverse_mu = jnp.stack(
                [read_beam._mu[..., 0], read_beam._mu[..., 2]], dim=-1
            )
            transverse_cov = jnp.stack(
                [
                    jnp.stack(
                        [read_beam._cov[..., 0, 0], read_beam._cov[..., 0, 2]], dim=-1
                    ),
                    jnp.stack(
                        [read_beam._cov[..., 2, 0], read_beam._cov[..., 2, 2]], dim=-1
                    ),
                ],
                dim=-1,
            )
            dist = MultivariateNormal(
                loc=transverse_mu, covariance_matrix=transverse_cov
            )

            left = self.extent[0]
            right = self.extent[1]
            hstep = self.pixel_size[0] * self.binning
            bottom = self.extent[2]
            top = self.extent[3]
            vstep = self.pixel_size[1] * self.binning
            x, y = jnp.meshgrid(
                jnp.arange(left, right, hstep),
                jnp.arange(bottom, top, vstep),
                indexing="ij",
            )
            pos = jnp.dstack((x, y))
            image = dist.log_prob(pos).exp()
            image = jnp.flip(image, dims=[1])
        elif isinstance(read_beam, ParticleBeam):
            if self.method == "histogram":
                # Catch vectorisation, which is currently not supported by "histogram"
                if (
                    len(read_beam.particles.shape) > 2
                    or len(read_beam.particle_charges.shape) > 1
                    or len(read_beam.energy.shape) > 0
                ):
                    raise NotImplementedError(
                        "The `'histogram'` method of `Screen` does not support "
                        "vectorization. Use `'kde'` instead. If this is a feature you "
                        "would like to see, please open an issue on GitHub."
                    )

                image, _ = jnp.histogramdd(
                    jnp.stack((read_beam.x, read_beam.y)).T,
                    bins=self.pixel_bin_edges,
                    weight=read_beam.particle_charges
                    * read_beam.survival_probabilities,
                )
                image = jnp.flipud(image.T)
            elif self.method == "kde":
                weights = read_beam.particle_charges * read_beam.survival_probabilities
                broadcasted_x, broadcasted_y, broadcasted_weights = (
                    jnp.broadcast_tensors(read_beam.x, read_beam.y, weights)
                )
                image = kde_histogram_2d(
                    x1=broadcasted_x,
                    x2=broadcasted_y,
                    bins1=self.pixel_bin_centers[0],
                    bins2=self.pixel_bin_centers[1],
                    bandwidth=self.kde_bandwidth,
                    weights=broadcasted_weights,
                )
                # Change the x, y positions
                image = jnp.transpose(image, -2, -1)
                # Flip up and down, now row 0 corresponds to the top
                image = jnp.flip(image, dims=[-2])
        else:
            raise TypeError(f"Read beam is of invalid type {type(read_beam)}")

        self.cached_reading = image
        return image

    def get_read_beam(self) -> Beam:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        return self._read_beam

    def set_read_beam(self, value: Beam) -> None:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        self._read_beam = value
        self.cached_reading = jnp.full(
            (self.resolution[0], self.resolution[1]),
            jnp.nan,
            device=self.cached_reading.device,
            dtype=self.cached_reading.dtype,
        )

    def split(self, resolution: jax.Array) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s

        alpha = 1 if self.is_active else 0.2

        patch = Rectangle(
            (plot_s, -0.6), 0, 0.6 * 2, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "resolution",
            "pixel_size",
            "binning",
            "misalignment",
            "method",
            "kde_bandwidth",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(resolution={repr(self.resolution)}, "
            + f"pixel_size={repr(self.pixel_size)}, "
            + f"binning={repr(self.binning)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"method={repr(self.method)}, "
            + f"kde_bandwidth={repr(self.kde_bandwidth)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
