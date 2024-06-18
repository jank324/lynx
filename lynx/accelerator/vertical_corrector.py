from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants

from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class VerticalCorrector(Element):
    """
    Verticle corrector magnet in a particle accelerator.
    Note: This is modeled as a drift section with
        a thin-kick in the vertical plane.

    :param length: Length in meters.
    :param angle: Particle deflection angle in the vertical plane in rad.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: jax.Array,
        angle: Optional[jax.Array] = None,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = jnp.asarray(length, **factory_kwargs)
        self.angle = (
            jnp.asarray(angle, **factory_kwargs)
            if angle is not None
            else jnp.zeros_like(self.length)
        )

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        device = self.length.device
        dtype = self.length.dtype

        gamma = energy / rest_energy.to(device=device, dtype=dtype)
        igamma2 = jnp.zeros_like(gamma)  # TODO: Effect on gradients?
        igamma2[gamma != 0] = 1 / gamma[gamma != 0] ** 2
        beta = jnp.sqrt(1 - igamma2)

        tm = jnp.eye(7, device=device, dtype=dtype).repeat((*self.length.shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 3, 6] = self.angle
        tm[..., 4, 5] = -self.length / beta**2 * igamma2
        return tm

    def broadcast(self, shape: tuple) -> Element:
        return self.__class__(
            length=self.length.repeat(shape), angle=self.angle, name=self.name
        )

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return any(self.angle != 0)

    def split(self, resolution: jax.Array) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            length = jnp.min(resolution, remaining)
            element = VerticalCorrector(length, self.angle * length / self.length)
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle[0]) if self.is_active else 1)

        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:cyan", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "angle"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"name={repr(self.name)})"
        )
