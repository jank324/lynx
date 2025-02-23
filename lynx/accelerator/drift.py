from typing import Literal, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.constants import physical_constants

from lynx.accelerator.element import Element
from lynx.particles import Beam, ParticleBeam
from lynx.utils import UniqueNameGenerator, bmadx, compute_relativistic_factors

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Drift(Element):
    """
    Drift section in a particle accelerator.

    NOTE: The transfer map now uses the linear approximation.
    Including the R_56 = L / (beta**2 * gamma **2)

    :param length: Length in meters.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: jnp.Array,
        tracking_method: Literal["lynx", "bmadx"] = "lynx",
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.length = jnp.as_tensor(length, **factory_kwargs)
        self.tracking_method = tracking_method

    def transfer_map(self, energy: jnp.Array) -> jnp.Array:
        device = self.length.device
        dtype = self.length.dtype

        _, igamma2, beta = compute_relativistic_factors(energy)

        vector_shape = jnp.broadcast_shapes(self.length.shape, igamma2.shape)

        tm = jnp.eye(7, device=device, dtype=dtype).repeat((*vector_shape, 1, 1))
        tm[..., 0, 1] = self.length
        tm[..., 2, 3] = self.length
        tm[..., 4, 5] = -self.length / beta**2 * igamma2

        return tm

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the dipole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "lynx":
            return super().track(incoming)
        elif self.tracking_method == "bmadx":
            assert isinstance(
                incoming, ParticleBeam
            ), "Bmad-X tracking is currently only supported for `ParticleBeam`."
            return self._track_bmadx(incoming)
        else:
            raise ValueError(
                f"Invalid tracking method {self.tracking_method}. "
                + "Supported methods are 'lynx' and 'bmadx'."
            )

    def _track_bmadx(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Track particles through the dipole element using the Bmad-X tracking method.

        :param incoming: Beam entering the element. Currently only supports
            `ParticleBeam`.
        :return: Beam exiting the element.
        """
        # Compute Bmad coordinates and p0c
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p

        z, pz, p0c = bmadx.lynx_to_bmad_z_pz(
            tau, delta, incoming.energy, electron_mass_eV
        )

        # Begin Bmad-X tracking
        x, y, z = bmadx.track_a_drift(
            self.length, x, px, y, py, z, pz, p0c, electron_mass_eV
        )
        # End of Bmad-X tracking

        # Convert back to Lynx coordinates
        tau, delta, ref_energy = bmadx.bmad_to_lynx_z_pz(z, pz, p0c, electron_mass_eV)

        # Broadcast to align their shapes so that they can be stacked
        x, px, y, py, tau, delta = jnp.broadcast_tensors(x, px, y, py, tau, delta)

        outgoing_beam = ParticleBeam(
            particles=jnp.stack(
                [x, px, y, py, tau, delta, jnp.ones_like(x)], dim=-1
            ),
            energy=ref_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            device=incoming.particles.device,
            dtype=incoming.particles.dtype,
        )
        return outgoing_beam

    @property
    def is_skippable(self) -> bool:
        return self.tracking_method == "lynx"

    def split(self, resolution: jnp.Telynx> list[Element]:
        num_splits = jnp.ceil(jnp.max(self.length) / resolution).int()
        return [
            Drift(
                self.length / num_splits,
                tracking_method=self.tracking_method,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            for i in range(num_splits)
        ]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        pass

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "tracking_method"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
