from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import constants
from scipy.constants import physical_constants

from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Drift(Element):
    """
    Drift section in a particle accelerator.

    Note: the transfer map now uses the linear approximation.
    Including the R_56 = L / (beta**2 * gamma **2)

    :param length: Length in meters.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: jax.Array,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = jnp.asarray(length, **factory_kwargs)

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        assert (
            energy.shape == self.length.shape
        ), f"Beam shape {energy.shape} does not match element shape {self.length.shape}"

        device = self.length.device
        dtype = self.length.dtype

        gamma = energy / rest_energy.to(device=device, dtype=dtype)
        igamma2 = jnp.zeros_like(gamma)  # TODO: Effect on gradients?
        igamma2[gamma != 0] = 1 / gamma[gamma != 0] ** 2
        beta = jnp.sqrt(1 - igamma2)

        tm = jnp.eye(7, device=device, dtype=dtype).repeat((*self.length.shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    def broadcast(self, shape: tuple) -> Element:
        return self.__class__(length=self.length.repeat(shape), name=self.name)

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: jax.Array) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Drift(jnp.min(resolution, remaining))
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: plt.Axes, s: float) -> None:
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={repr(self.length)})"
