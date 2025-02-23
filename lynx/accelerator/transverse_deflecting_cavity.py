from typing import Literal, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.constants import physical_constants, speed_of_light

from lynx.accelerator.element import Element
from lynx.particles import Beam, ParticleBeam
from lynx.utils import UniqueNameGenerator, bmadx, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class TransverseDeflectingCavity(Element):
    """
    Transverse deflecting cavity element.

    :param length: Length in meters.
    :param voltage: Voltage of the cavity in volts.
    :param phase: Phase of the cavity in radians.
    :param frequency: Frequency of the cavity in Hz.
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
        voltage: Optional[jnp.Array] = None,
        phase: Optional[jnp.Array] = None,
        frequency: Optional[jnp.Array] = None,
        misalignment: Optional[jnp.Array] = None,
        tilt: Optional[jnp.Array] = None,
        num_steps: int = 1,
        tracking_method: Literal["bmadx"] = "bmadx",
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [length, voltage, phase, frequency, misalignment, tilt], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.register_buffer("voltage", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("phase", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("frequency", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("misalignment", jnp.asarray((0.0, 0.0), **factory_kwargs))
        self.register_buffer("tilt", jnp.asarray(0.0, **factory_kwargs))

        self.length = jnp.as_tensor(length, **factory_kwargs)
        if voltage is not None:
            self.voltage = jnp.as_tensor(voltage, **factory_kwargs)
        if phase is not None:
            self.phase = jnp.as_tensor(phase, **factory_kwargs)
        if frequency is not None:
            self.frequency = jnp.as_tensor(frequency, **factory_kwargs)
        if misalignment is not None:
            self.misalignment = jnp.as_tensor(misalignment, **factory_kwargs)
        if tilt is not None:
            self.tilt = jnp.as_tensor(tilt, **factory_kwargs)

        self.num_steps = num_steps
        self.tracking_method = tracking_method

    @property
    def is_active(self) -> bool:
        return jnp.any(self.voltage != 0)

    @property
    def is_skippable(self) -> bool:
        # TODO: Implement drrift-like `transfer_map` and set to `self.is_active`
        return False

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the transverse deflecting cavity.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "lynx":
            raise NotImplementedError(
                "Lynx transverse deflecting cavity tracking is not yet implemented."
            )
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
        Track particles through the TDC element using the Bmad-X tracking method.

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

        # Begin Bmad-X tracking
        x, px, y, py = bmadx.offset_particle_set(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        x, y, z = bmadx.track_a_drift(
            self.length / 2, x, px, y, py, z, pz, p0c, electron_mass_eV
        )

        voltage = self.voltage / p0c
        k_rf = 2 * jnp.pi * self.frequency / speed_of_light
        # Phase that the particle sees
        phase = (
            2
            * jnp.pi
            * (
                self.phase.unsqueeze(-1)
                - (
                    bmadx.particle_rf_time(z, pz, p0c, electron_mass_eV)
                    * self.frequency.unsqueeze(-1)
                )
            )
        )

        # TODO: Assigning px to px is really bad practice and should be separated into
        # two separate variables
        px = px + voltage.unsqueeze(-1) * jnp.sin(phase)

        beta_old = (
            (1 + pz)
            * p0c.unsqueeze(-1)
            / jnp.sqrt(((1 + pz) * p0c.unsqueeze(-1)) ** 2 + electron_mass_eV**2)
        )
        E_old = (1 + pz) * p0c.unsqueeze(-1) / beta_old
        E_new = E_old + voltage.unsqueeze(-1) * jnp.cos(phase) * k_rf.unsqueeze(
            -1
        ) * x * p0c.unsqueeze(-1)
        pc = jnp.sqrt(E_new**2 - electron_mass_eV**2)
        beta = pc / E_new

        pz = (pc - p0c.unsqueeze(-1)) / p0c.unsqueeze(-1)
        z = z * beta / beta_old

        x, y, z = bmadx.track_a_drift(
            self.length / 2, x, px, y, py, z, pz, p0c, electron_mass_eV
        )

        x, px, y, py = bmadx.offset_particle_unset(
            x_offset, y_offset, self.tilt, x, px, y, py
        )
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

    def split(self, resolution: jnp.Array) -> list[Element]:
        # TODO: Implement splitting for cavity properly, for now just returns the
        # element itself
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        height = 0.4

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="olive", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "voltage",
            "phase",
            "frequency",
            "misalignment",
            "tilt",
            "num_steps",
            "tracking_method",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"voltage={repr(self.voltage)}, "
            + f"phase={repr(self.phase)}, "
            + f"frequency={repr(self.frequency)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"num_steps={repr(self.num_steps)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
