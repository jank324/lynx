from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants

from lynx.track_methods import base_rmatrix, rotation_matrix
from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Dipole(Element):
    """
    Dipole magnet (by default a sector bending magnet).

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param e1: The angle of inclination of the entrance face [rad].
    :param e2: The angle of inclination of the exit face [rad].
    :param tilt: Tilt of the magnet in x-y plane [rad].
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: (only set if different from `fint`) Fringe field
        integral of the exit face.
    :param gap: The magnet gap [m], NOTE in MAD and ELEGANT: HGAP = gap/2
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: Union[jax.Array, nn.Parameter],
        angle: Optional[Union[jax.Array, nn.Parameter]] = None,
        e1: Optional[Union[jax.Array, nn.Parameter]] = None,
        e2: Optional[Union[jax.Array, nn.Parameter]] = None,
        tilt: Optional[Union[jax.Array, nn.Parameter]] = None,
        fringe_integral: Optional[Union[jax.Array, nn.Parameter]] = None,
        fringe_integral_exit: Optional[Union[jax.Array, nn.Parameter]] = None,
        gap: Optional[Union[jax.Array, nn.Parameter]] = None,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.length = jnp.asarray(length, **factory_kwargs)
        self.angle = (
            jnp.asarray(angle, **factory_kwargs)
            if angle is not None
            else jnp.zeros_like(self.length)
        )
        self.gap = (
            jnp.asarray(gap, **factory_kwargs)
            if gap is not None
            else jnp.zeros_like(self.length)
        )
        self.tilt = (
            jnp.asarray(tilt, **factory_kwargs)
            if tilt is not None
            else jnp.zeros_like(self.length)
        )
        self.name = name
        self.fringe_integral = (
            jnp.asarray(fringe_integral, **factory_kwargs)
            if fringe_integral is not None
            else jnp.zeros_like(self.length)
        )
        self.fringe_integral_exit = (
            self.fringe_integral
            if fringe_integral_exit is None
            else jnp.asarray(fringe_integral_exit, **factory_kwargs)
        )
        # Sector bend if not specified
        self.e1 = (
            jnp.asarray(e1, **factory_kwargs)
            if e1 is not None
            else jnp.zeros_like(self.length)
        )
        self.e2 = (
            jnp.asarray(e2, **factory_kwargs)
            if e2 is not None
            else jnp.zeros_like(self.length)
        )

    @property
    def hx(self) -> jax.Array:
        value = jnp.zeros_like(self.length)
        value[self.length != 0] = (
            self.angle[self.length != 0] / self.length[self.length != 0]
        )
        return value

    @property
    def is_skippable(self) -> bool:
        return True

    @property
    def is_active(self):
        return any(self.angle != 0)

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        device = self.length.device
        dtype = self.length.dtype

        R_enter = self._transfer_map_enter()
        R_exit = self._transfer_map_exit()

        if any(self.length != 0.0):  # Bending magnet with finite length
            R = base_rmatrix(
                length=self.length,
                k1=jnp.zeros_like(self.length),
                hx=self.hx,
                tilt=jnp.zeros_like(self.length),
                energy=energy,
            )  # Tilt is applied after adding edges
        else:  # Reduce to Thin-Corrector
            R = jnp.eye(7, device=device, dtype=dtype).repeat(
                (*self.length.shape, 1, 1)
            )
            R[..., 0, 1] = self.length
            R[..., 2, 6] = self.angle
            R[..., 2, 3] = self.length

        # Apply fringe fields
        R = jnp.matmul(R_exit, jnp.matmul(R, R_enter))
        # Apply rotation for tilted magnets
        R = jnp.matmul(
            rotation_matrix(-self.tilt), jnp.matmul(R, rotation_matrix(self.tilt))
        )
        return R

    def _transfer_map_enter(self) -> jax.Array:
        """Linear transfer map for the entrance face of the dipole magnet."""
        device = self.length.device
        dtype = self.length.dtype

        sec_e = 1.0 / jnp.cos(self.e1)
        phi = (
            self.fringe_integral
            * self.hx
            * self.gap
            * sec_e
            * (1 + jnp.sin(self.e1) ** 2)
        )

        tm = jnp.eye(7, device=device, dtype=dtype).repeat(*phi.shape, 1, 1)
        tm[..., 1, 0] = self.hx * jnp.tan(self.e1)
        tm[..., 3, 2] = -self.hx * jnp.tan(self.e1 - phi)

        return tm

    def _transfer_map_exit(self) -> jax.Array:
        """Linear transfer map for the exit face of the dipole magnet."""
        device = self.length.device
        dtype = self.length.dtype

        sec_e = 1.0 / jnp.cos(self.e2)
        phi = (
            self.fringe_integral_exit
            * self.hx
            * self.gap
            * sec_e
            * (1 + jnp.sin(self.e2) ** 2)
        )

        tm = jnp.eye(7, device=device, dtype=dtype).repeat(*phi.shape, 1, 1)
        tm[..., 1, 0] = self.hx * jnp.tan(self.e2)
        tm[..., 3, 2] = -self.hx * jnp.tan(self.e2 - phi)

        return tm

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            length=self.length.repeat(shape),
            angle=self.angle.repeat(shape),
            e1=self.e1.repeat(shape),
            e2=self.e2.repeat(shape),
            tilt=self.tilt.repeat(shape),
            fringe_integral=self.fringe_integral.repeat(shape),
            fringe_integral_exit=self.fringe_integral_exit.repeat(shape),
            gap=self.gap.repeat(shape),
            name=self.name,
        )

    def split(self, resolution: jax.Array) -> list[Element]:
        # TODO: Implement splitting for dipole properly, for now just returns the
        # element itself
        return [self]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"e1={repr(self.e1)},"
            + f"e2={repr(self.e2)},"
            + f"tilt={repr(self.tilt)},"
            + f"fringe_integral={repr(self.fringe_integral)},"
            + f"fringe_integral_exit={repr(self.fringe_integral_exit)},"
            + f"gap={repr(self.gap)},"
            + f"name={repr(self.name)})"
        )

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "angle",
            "e1",
            "e2",
            "tilt",
            "fringe_integral",
            "fringe_integral_exit",
            "gap",
        ]

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (np.sign(self.angle[0]) if self.is_active else 1)

        patch = Rectangle(
            (s, 0), self.length[0], height, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)
