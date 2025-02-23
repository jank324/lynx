from typing import Literal, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.constants import physical_constants

from lynx.accelerator.element import Element
from lynx.particles import Beam, ParticleBeam
from lynx.track_methods import base_rmatrix, misalignment_matrix
from lynx.utils import UniqueNameGenerator, bmadx, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Quadrupole(Element):
    """
    Quadrupole magnet in a particle accelerator.

    :param length: Length in meters.
    :param k1: Strength of the quadrupole in 1/m^-2.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane [rad]. pi/4 for
        skew-quadrupole.
    :param num_steps: Number of drift-kick-drift steps to use for tracking through the
        element when tracking method is set to `"bmadx"`.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: jnp.Array,
        k1: Optional[jnp.Array] = None,
        misalignment: Optional[jnp.Array] = None,
        tilt: Optional[jnp.Array] = None,
        num_steps: int = 1,
        tracking_method: Literal["lynx", "bmadx"] = "lynx",
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [length, k1, misalignment, tilt], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.register_buffer("k1", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("misalignment", jnp.asarray((0.0, 0.0), **factory_kwargs))
        self.register_buffer("tilt", jnp.asarray(0.0, **factory_kwargs))

        self.length = jnp.as_tensor(length, **factory_kwargs)
        if k1 is not None:
            self.k1 = jnp.as_tensor(k1, **factory_kwargs)
        if misalignment is not None:
            self.misalignment = jnp.as_tensor(misalignment, **factory_kwargs)
        if tilt is not None:
            self.tilt = jnp.as_tensor(tilt, **factory_kwargs)

        self.num_steps = num_steps
        self.tracking_method = tracking_method

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

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the quadrupole element.

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
        Track particles through the quadrupole element using the Bmad-X tracking method.

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

        x_offset = self.misalignment[..., 0]
        y_offset = self.misalignment[..., 1]

        step_length = self.length / self.num_steps
        b1 = self.k1 * self.length

        # Begin Bmad-X tracking
        x, px, y, py = bmadx.offset_particle_set(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        for _ in range(self.num_steps):
            rel_p = 1 + pz  # Particle's relative momentum (P/P0)
            k1 = b1.unsqueeze(-1) / (self.length.unsqueeze(-1) * rel_p)

            tx, dzx = bmadx.calculate_quadrupole_coefficients(-k1, step_length, rel_p)
            ty, dzy = bmadx.calculate_quadrupole_coefficients(k1, step_length, rel_p)

            z = (
                z
                + dzx[0] * x**2
                + dzx[1] * x * px
                + dzx[2] * px**2
                + dzy[0] * y**2
                + dzy[1] * y * py
                + dzy[2] * py**2
            )

            x_next = tx[0][0] * x + tx[0][1] * px
            px_next = tx[1][0] * x + tx[1][1] * px
            y_next = ty[0][0] * y + ty[0][1] * py
            py_next = ty[1][0] * y + ty[1][1] * py

            x, px, y, py = x_next, px_next, y_next, py_next

            z = z + bmadx.low_energy_z_correction(
                pz, p0c, electron_mass_eV, step_length
            )

        # s = s + l
        x, px, y, py = bmadx.offset_particle_unset(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        # pz is unaffected by tracking, therefore needs to match vector dimensions
        pz = pz * jnp.ones_like(x)
        # End of Bmad-X tracking

        # Convert back to Lynx coordinates
        tau, delta, ref_energy = bmadx.bmad_to_lynx_z_pz(z, pz, p0c, electron_mass_eV)

        outgoing_beam = ParticleBeam(
            particles=jnp.stack((x, px, y, py, tau, delta, jnp.ones_like(x)), dim=-1),
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

    @property
    def is_active(self) -> bool:
        return jnp.any(self.k1 != 0)

    def split(self, resolution: jnp.Array) -> list[Element]:
        num_splits = jnp.ceil(jnp.max(self.length) / resolution).int()
        return [
            Quadrupole(
                self.length / num_splits,
                self.k1,
                misalignment=self.misalignment,
                tilt=self.tilt,
                num_steps=self.num_steps,
                tracking_method=self.tracking_method,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            for i in range(num_splits)
        ]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_k1 = self.k1[vector_idx] if self.k1.dim() > 0 else self.k1
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (jnp.sign(plot_k1) if self.is_active else 1)
        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:red", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length", "k1", "misalignment", "tilt"]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"k1={repr(self.k1)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"num_steps={repr(self.num_steps)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
