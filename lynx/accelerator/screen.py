from copy import deepcopy
from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants
from torch.distributions import MultivariateNormal

from lynx.particles import Beam, ParameterBeam, ParticleBeam
from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Screen(Element):
    """
    Diagnostic screen in a particle accelerator.

    :param resolution: Resolution of the camera sensor looking at the screen given as
        Tensor `(width, height)`.
    :param pixel_size: Size of a pixel on the screen in meters given as a Tensor
        `(width, height)`.
    :param binning: Binning used by the camera.
    :param misalignment: Misalignment of the screen in meters given as a Tensor
        `(x, y)`.
    :param is_active: If `True` the screen is active and will record the beam's
        distribution. If `False` the screen is inactive and will not record the beam's
        distribution.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        resolution: Optional[Union[jax.Array, nn.Parameter]] = None,
        pixel_size: Optional[Union[jax.Array, nn.Parameter]] = None,
        binning: Optional[Union[jax.Array, nn.Parameter]] = None,
        misalignment: Optional[Union[jax.Array, nn.Parameter]] = None,
        is_active: bool = False,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.resolution = (
            jnp.asarray(resolution, **factory_kwargs)
            if resolution is not None
            else jnp.array((1024, 1024), **factory_kwargs)
        )
        self.pixel_size = (
            jnp.asarray(pixel_size, **factory_kwargs)
            if pixel_size is not None
            else jnp.array((1e-3, 1e-3), **factory_kwargs)
        )
        self.binning = (
            jnp.asarray(binning, **factory_kwargs)
            if binning is not None
            else jnp.array(1, **factory_kwargs)
        )
        self.misalignment = (
            jnp.asarray(misalignment, **factory_kwargs)
            if misalignment is not None
            else jnp.array((0.0, 0.0), **factory_kwargs)
        )
        self.length = jnp.zeros(self.misalignment.shape[:-1], **factory_kwargs)
        self.is_active = is_active

        self.set_read_beam(None)
        self.cached_reading = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    @property
    def effective_resolution(self) -> jax.Array:
        return self.resolution / self.binning

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

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        device = self.misalignment.device
        dtype = self.misalignment.dtype

        return jnp.eye(7, device=device, dtype=dtype).repeat((*energy.shape, 1, 1))

    def track(self, incoming: Beam) -> Beam:
        if self.is_active:
            copy_of_incoming = deepcopy(incoming)

            if isinstance(incoming, ParameterBeam):
                copy_of_incoming._mu[:, 0] -= self.misalignment[:, 0]
                copy_of_incoming._mu[:, 2] -= self.misalignment[:, 1]
            elif isinstance(incoming, ParticleBeam):
                copy_of_incoming.particles[:, :, 0] -= self.misalignment[:, 0]
                copy_of_incoming.particles[:, :, 1] -= self.misalignment[:, 1]

            self.set_read_beam(copy_of_incoming)

            return Beam.empty
        else:
            return incoming

    @property
    def reading(self) -> jax.Array:
        if self.cached_reading is not None:
            return self.cached_reading

        read_beam = self.get_read_beam()
        if read_beam is Beam.empty or read_beam is None:
            image = jnp.zeros(
                (
                    *self.misalignment.shape[:-1],
                    int(self.effective_resolution[1]),
                    int(self.effective_resolution[0]),
                )
            )
        elif isinstance(read_beam, ParameterBeam):
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
            dist = [
                MultivariateNormal(
                    loc=transverse_mu_sample, covariance_matrix=transverse_cov_sample
                )
                for transverse_mu_sample, transverse_cov_sample in zip(
                    transverse_mu.cpu(), transverse_cov.cpu()
                )
            ]

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
            image = jnp.stack([dist_sample.log_prob(pos).exp() for dist_sample in dist])
            image = jnp.flip(image, dims=[1])
        elif isinstance(read_beam, ParticleBeam):
            image = jnp.zeros(
                (
                    *self.misalignment.shape[:-1],
                    int(self.effective_resolution[1]),
                    int(self.effective_resolution[0]),
                )
            )
            for i, (xs_sample, ys_sample) in enumerate(zip(read_beam.xs, read_beam.ys)):
                image_sample, _ = jnp.histogramdd(
                    jnp.stack((xs_sample, ys_sample)).T.cpu(),
                    bins=self.pixel_bin_edges,
                )
                image_sample = jnp.flipud(image_sample.T)
                image_sample = image_sample.cpu()

                image[i] = image_sample
        else:
            raise TypeError(f"Read beam is of invalid type {type(read_beam)}")

        self.cached_reading = image
        return image

    def get_read_beam(self) -> Beam:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        return self._read_beam[0] if self._read_beam is not None else None

    def set_read_beam(self, value: Beam) -> None:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        self._read_beam = [value]
        self.cached_reading = None

    def broadcast(self, shape: Size) -> Element:
        new_screen = self.__class__(
            resolution=self.resolution,
            pixel_size=self.pixel_size,
            binning=self.binning,
            misalignment=self.misalignment.repeat((*shape, 1)),
            is_active=self.is_active,
            name=self.name,
        )
        new_screen.length = self.length.repeat(shape)
        return new_screen

    def split(self, resolution: jax.Array) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle(
            (s, -0.6), 0, 0.6 * 2, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "resolution",
            "pixel_size",
            "binning",
            "misalignment",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(resolution={repr(self.resolution)}, "
            + f"pixel_size={repr(self.pixel_size)}, "
            + f"binning={repr(self.binning)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
