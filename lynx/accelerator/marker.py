from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import constants
from scipy.constants import physical_constants

from lynx.particles import Beam
from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Marker(Element):
    """
    General Marker / Monitor element

    :param name: Unique identifier of the element.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        return jnp.eye(7, device=energy.device, dtype=energy.dtype).repeat(
            (*energy.shape, 1, 1)
        )

    def track(self, incoming: Beam) -> Beam:
        # TODO: At some point Markers should be able to be active or inactive. Active
        # Markers would be able to record the beam tracked through them.
        return incoming

    def broadcast(self, shape: tuple) -> Element:
        new_marker = self.__class__(name=self.name)
        new_marker.length = self.length.repeat(shape)
        return new_marker

    @property
    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: jax.Array) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        # Do nothing on purpose. Maybe later we decide markers should be shown, but for
        # now they are invisible.
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={repr(self.name)})"
