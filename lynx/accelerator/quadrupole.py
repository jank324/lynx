from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants

from lynx.track_methods import base_rmatrix, misalignment_matrix
from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Quadrupole(Element):
    """
    Quadrupole magnet in a particle accelerator.

    :param length: Length in meters.
    :param k1: Strength of the quadrupole in rad/m.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane [rad]. pi/4 for
        skew-quadrupole.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: jax.Array,
        k1: Optional[jax.Array] = None,
        misalignment: Optional[jax.Array] = None,
        tilt: Optional[jax.Array] = None,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = jnp.asarray(length, **factory_kwargs)
        self.k1 = (
            jnp.asarray(k1, **factory_kwargs)
            if k1 is not None
            else jnp.zeros_like(self.length)
        )
        self.misalignment = (
            jnp.asarray(misalignment, **factory_kwargs)
            if misalignment is not None
            else jnp.zeros((*self.length.shape, 2), **factory_kwargs)
        )
        self.tilt = (
            jnp.asarray(tilt, **factory_kwargs)
            if tilt is not None
            else jnp.zeros_like(self.length)
        )

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        R = base_rmatrix(
            length=self.length,
            k1=self.k1,
            hx=jnp.zeros_like(self.length),
            tilt=self.tilt,
            energy=energy,
        )

        if jnp.all(self.misalignment == 0):
            return R
        else:
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            R = jnp.einsum("...ij,...jk,...kl->...il", R_exit, R, R_entry)
            return R

    def broadcast(self, shape: tuple) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            k1=self.k1.repeat(shape),
            misalignment=self.misalignment.repeat((*shape, 1)),
            tilt=self.tilt.repeat(shape),
            name=self.name,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return any(self.k1 != 0)

    def split(self, resolution: jax.Array) -> list[Element]:
        split_elements = []
        remaining = self.length
        while remaining > 0:
            element = Quadrupole(
                jnp.min(resolution, remaining),
                self.k1,
                misalignment=self.misalignment,
            )
            split_elements.append(element)
            remaining -= resolution
        return split_elements

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.k1[0]) if self.is_active else 1)
        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:red", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k1", "misalignment", "tilt"]

    def __repr__(self) -> None:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"k1={repr(self.k1)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"name={repr(self.name)})"
        )
