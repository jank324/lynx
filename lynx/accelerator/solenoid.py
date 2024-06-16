from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants

from lynx.track_methods import misalignment_matrix
from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Solenoid(Element):
    """
    Solenoid magnet.

    Implemented according to A.W.Chao P74

    :param length: Length in meters.
    :param k: Normalised strength of the solenoid magnet B0/(2*Brho). B0 is the field
        inside the solenoid, Brho is the momentum of central trajectory.
    :param misalignment: Misalignment vector of the solenoid magnet in x- and
        y-directions.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[jax.Array, nn.Parameter] = None,
        k: Optional[Union[jax.Array, nn.Parameter]] = None,
        misalignment: Optional[Union[jax.Array, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = jnp.asarray(length, **factory_kwargs)
        self.k = (
            jnp.asarray(k, **factory_kwargs)
            if k is not None
            else jnp.zeros_like(self.length)
        )
        self.misalignment = (
            jnp.asarray(misalignment, **factory_kwargs)
            if misalignment is not None
            else jnp.zeros((*self.length.shape[:-1], 2), **factory_kwargs)
        )

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        device = self.length.device
        dtype = self.length.dtype

        gamma = energy / rest_energy.to(device=device, dtype=dtype)
        c = jnp.cos(self.length * self.k)
        s = jnp.sin(self.length * self.k)

        s_k = jnp.empty_like(self.length)
        s_k[self.k == 0] = self.length[self.k == 0]
        s_k[self.k != 0] = s[self.k != 0] / self.k[self.k != 0]

        r56 = jnp.zeros_like(self.length)
        if gamma != 0:
            gamma2 = gamma * gamma
            beta = jnp.sqrt(1.0 - 1.0 / gamma2)
            r56 -= self.length / (beta * beta * gamma2)

        R = jnp.eye(7, device=device, dtype=dtype).repeat((*self.length.shape, 1, 1))
        R[..., 0, 0] = c**2
        R[..., 0, 1] = c * s_k
        R[..., 0, 2] = s * c
        R[..., 0, 3] = s * s_k
        R[..., 1, 0] = -self.k * s * c
        R[..., 1, 1] = c**2
        R[..., 1, 2] = -self.k * s**2
        R[..., 1, 3] = s * c
        R[..., 2, 0] = -s * c
        R[..., 2, 1] = -s * s_k
        R[..., 2, 2] = c**2
        R[..., 2, 3] = c * s_k
        R[..., 3, 0] = self.k * s**2
        R[..., 3, 1] = -s * c
        R[..., 3, 2] = -self.k * s * c
        R[..., 3, 3] = c**2
        R[..., 4, 5] = r56

        R = R.real

        if jnp.all(self.misalignment == 0):
            return R
        else:
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            R = jnp.einsum("...ij,...jk,...kl->...il", R_exit, R, R_entry)
            return R

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            k=self.k.repeat(shape),
            misalignment=self.misalignment.repeat(shape),
            name=self.name,
        )

    @property
    def is_active(self) -> bool:
        return any(self.k != 0)

    def is_skippable(self) -> bool:
        return True

    def split(self, resolution: jax.Array) -> list[Element]:
        # TODO: Implement splitting for solenoid properly, for now just return self
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8

        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:orange", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k", "misalignment"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"k={repr(self.k)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"name={repr(self.name)})"
        )
