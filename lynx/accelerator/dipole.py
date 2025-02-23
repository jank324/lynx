from typing import Literal, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.constants import physical_constants

from lynx.accelerator.element import Element
from lynx.particles import Beam, ParticleBeam
from lynx.track_methods import base_rmatrix, rotation_matrix
from lynx.utils import UniqueNameGenerator, bmadx, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Dipole(Element):
    """
    Dipole magnet (by default a sector bending magnet).

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param k1: Focussing strength in 1/m^-2. Only used with `"lynx"` tracking method.
    :param dipole_e1: The angle of inclination of the entrance face in rad.
    :param dipole_e2: The angle of inclination of the exit face in rad.
    :param tilt: Tilt of the magnet in x-y plane [rad].
    :param gap: The magnet gap in meters. Note that in MAD and ELEGANT: HGAP = gap/2.
    :param gap_exit: The magnet gap at the exit in meters. Note that in MAD and
        ELEGANT: HGAP = gap/2. Only set if different from `gap`. Only used with
        `"bmadx"` tracking method.
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: Fringe field integral of the exit face. Only set if
        different from `fringe_integral`. Only used with `"bmadx"` tracking method.
    :param fringe_at: Where to apply the fringe fields for `"bmadx"` tracking. The
        available options are:
        - "neither": Do not apply fringe fields.
        - "entrance": Apply fringe fields at the entrance end.
        - "exit": Apply fringe fields at the exit end.
        - "both": Apply fringe fields at both ends.
    :param fringe_type: Type of fringe field for `"bmadx"` tracking. Currently only
        supports `"linear_edge"`.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: jnp.Array,
        angle: Optional[jnp.Array] = None,
        k1: Optional[jnp.Array] = None,
        dipole_e1: Optional[jnp.Array] = None,
        dipole_e2: Optional[jnp.Array] = None,
        tilt: Optional[jnp.Array] = None,
        gap: Optional[jnp.Array] = None,
        gap_exit: Optional[jnp.Array] = None,
        fringe_integral: Optional[jnp.Array] = None,
        fringe_integral_exit: Optional[jnp.Array] = None,
        fringe_at: Literal["neither", "entrance", "exit", "both"] = "both",
        fringe_type: Literal["linear_edge"] = "linear_edge",
        tracking_method: Literal["lynx", "bmadx"] = "lynx",
        name: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        device, dtype = verify_device_and_dtype(
            [
                length,
                angle,
                k1,
                dipole_e1,
                dipole_e2,
                tilt,
                gap,
                gap_exit,
                fringe_integral,
                fringe_integral_exit,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.register_buffer("angle", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("k1", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("_e1", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("_e2", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("fringe_integral", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("fringe_integral_exit", None)
        self.register_buffer("gap", jnp.asarray(0.0, **factory_kwargs))
        self.register_buffer("gap_exit", None)
        self.register_buffer("tilt", jnp.asarray(0.0, **factory_kwargs))

        self.length = jnp.as_tensor(length, **factory_kwargs)
        if angle is not None:
            self.angle = jnp.as_tensor(angle, **factory_kwargs)
        if k1 is not None:
            self.k1 = jnp.as_tensor(k1, **factory_kwargs)
        if dipole_e1 is not None:
            self._e1 = jnp.as_tensor(dipole_e1, **factory_kwargs)
        if dipole_e2 is not None:
            self._e2 = jnp.as_tensor(dipole_e2, **factory_kwargs)
        if fringe_integral is not None:
            self.fringe_integral = jnp.as_tensor(fringe_integral, **factory_kwargs)
        self.fringe_integral_exit = (
            jnp.as_tensor(fringe_integral_exit, **factory_kwargs)
            if fringe_integral_exit is not None
            else self.fringe_integral
        )
        if gap is not None:
            self.gap = jnp.as_tensor(gap, **factory_kwargs)
        self.gap_exit = (
            jnp.as_tensor(gap_exit, **factory_kwargs)
            if gap_exit is not None
            else self.gap
        )
        if tilt is not None:
            self.tilt = jnp.as_tensor(tilt, **factory_kwargs)

        self.fringe_at = fringe_at
        self.fringe_type = fringe_type
        self.tracking_method = tracking_method

    @property
    def hx(self) -> jnp.Array:
        return jnp.where(self.length == 0.0, 0.0, self.angle / self.length)

    @property
    def dipole_e1(self) -> jnp.Array:
        return self._e1

    @dipole_e1.setter
    def dipole_e1(self, value: jnp.Array):
        self._e1 = value

    @property
    def dipole_e2(self) -> jnp.Array:
        return self._e2

    @dipole_e2.setter
    def dipole_e2(self, value: jnp.Array):
        self._e2 = value

    @property
    def is_skippable(self) -> bool:
        return self.tracking_method == "lynx"

    @property
    def is_active(self):
        return jnp.any(self.angle != 0)

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
        # TODO: The renaming of the compinents of `incoming` to just the component name
        # makes things hard to read. The resuse and overwriting of those component names
        # throughout the function makes it even hard, is bad practice and should really
        # be fixed!

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
        x, px, y, py = bmadx.offset_particle_set(
            jnp.zeros_like(self.tilt),
            jnp.zeros_like(self.tilt),
            self.tilt,
            x,
            px,
            y,
            py,
        )

        if self.fringe_at == "entrance" or self.fringe_at == "both":
            px, py = self._bmadx_fringe_linear("entrance", x, px, y, py)
        x, px, y, py, z, pz = self._bmadx_body(
            x, px, y, py, z, pz, p0c, electron_mass_eV
        )
        if self.fringe_at == "exit" or self.fringe_at == "both":
            px, py = self._bmadx_fringe_linear("exit", x, px, y, py)

        x, px, y, py = bmadx.offset_particle_unset(
            jnp.zeros_like(self.tilt),
            jnp.zeros_like(self.tilt),
            self.tilt,
            x,
            px,
            y,
            py,
        )
        # End of Bmad-X tracking

        # Convert back to Lynx coordinates
        tau, delta, ref_energy = bmadx.bmad_to_lynx_z_pz(z, pz, p0c, electron_mass_eV)

        # Broadcast to align their shapes so that they can be stacked
        x, px, y, py, tau, delta = jnp.broadcast_tensors(x, px, y, py, tau, delta)

        outgoing_beam = ParticleBeam(
            particles=jnp.stack((x, px, y, py, tau, delta, jnp.ones_like(x)), dim=-1),
            energy=ref_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            device=incoming.particles.device,
            dtype=incoming.particles.dtype,
        )
        return outgoing_beam

    def _bmadx_body(
        self,
        x: jnp.Array,
        px: jnp.Array,
        y: jnp.Array,
        py: jnp.Array,
        z: jnp.Array,
        pz: jnp.Array,
        p0c: jnp.Array,
        mc2: float,
    ) -> list[jnp.Array]:
        """
        Track particle coordinates through bend body.

        :param x: Initial x coordinate [m].
        :param px: Initial Bmad cannonical px coordinate.
        :param y: Initial y coordinate [m].
        :param py: Initial Bmad cannonical py coordinate.
        :param z: Initial Bmad cannonical z coordinate [m].
        :param pz: Initial Bmad cannonical pz coordinate.
        :param p0c: Reference momentum [eV/c].
        :param mc2: Particle mass [eV/c^2].
        :return: x, px, y, py, z, pz final Bmad cannonical coordinates.
        """
        px_norm = jnp.sqrt((1 + pz) ** 2 - py**2)  # For simplicity
        phi1 = jnp.arcsin(px / px_norm)
        g = self.angle / self.length
        gp = g.unsqueeze(-1) / px_norm

        alpha = (
            2
            * (1 + g.unsqueeze(-1) * x)
            * jnp.sin(self.angle.unsqueeze(-1) + phi1)
            * self.length.unsqueeze(-1)
            * bmadx.sinc(self.angle).unsqueeze(-1)
            - gp
            * (
                (1 + g.unsqueeze(-1) * x)
                * self.length.unsqueeze(-1)
                * bmadx.sinc(self.angle).unsqueeze(-1)
            )
            ** 2
        )

        x2_t1 = x * jnp.cos(self.angle.unsqueeze(-1)) + self.length.unsqueeze(
            -1
        ) ** 2 * g.unsqueeze(-1) * bmadx.cosc(self.angle.unsqueeze(-1))

        x2_t2 = jnp.sqrt((jnp.cos(self.angle.unsqueeze(-1) + phi1) ** 2) + gp * alpha)
        x2_t3 = jnp.cos(self.angle.unsqueeze(-1) + phi1)

        c1 = x2_t1 + alpha / (x2_t2 + x2_t3)
        c2 = x2_t1 + (x2_t2 - x2_t3) / gp
        temp = jnp.abs(self.angle.unsqueeze(-1) + phi1)
        x2 = c1 * (temp < jnp.pi / 2) + c2 * (temp >= jnp.pi / 2)

        Lcu = (
            x2
            - self.length.unsqueeze(-1) ** 2
            * g.unsqueeze(-1)
            * bmadx.cosc(self.angle.unsqueeze(-1))
            - x * jnp.cos(self.angle.unsqueeze(-1))
        )

        Lcv = -self.length.unsqueeze(-1) * bmadx.sinc(
            self.angle.unsqueeze(-1)
        ) - x * jnp.sin(self.angle.unsqueeze(-1))

        theta_p = 2 * (
            self.angle.unsqueeze(-1) + phi1 - jnp.pi / 2 - jnp.arctan2(Lcv, Lcu)
        )

        Lc = jnp.sqrt(Lcu**2 + Lcv**2)
        Lp = Lc / bmadx.sinc(theta_p / 2)

        P = p0c.unsqueeze(-1) * (1 + pz)  # In eV
        E = jnp.sqrt(P**2 + mc2**2)  # In eV
        E0 = jnp.sqrt(p0c**2 + mc2**2)  # In eV
        beta = P / E
        beta0 = p0c / E0

        x_f = x2
        px_f = px_norm * jnp.sin(self.angle.unsqueeze(-1) + phi1 - theta_p)
        y_f = y + py * Lp / px_norm
        z_f = (
            z
            + (beta * self.length.unsqueeze(-1) / beta0.unsqueeze(-1))
            - ((1 + pz) * Lp / px_norm)
        )

        return x_f, px_f, y_f, py, z_f, pz

    def _bmadx_fringe_linear(
        self,
        location: Literal["entrance", "exit"],
        x: jnp.Array,
        px: jnp.Array,
        y: jnp.Array,
        py: jnp.Array,
    ) -> list[jnp.Array]:
        """
        Tracks linear fringe.

        :param location: "entrance" or "exit".
        :param x: Initial x coordinate [m].
        :param px: Initial Bmad cannonical px coordinate.
        :param y: Initial y coordinate [m].
        :param py: Initial Bmad cannonical py coordinate.
        :return: px, py final Bmad cannonical coordinates.
        """
        g = self.angle / self.length
        e = self._e1 * (location == "entrance") + self._e2 * (location == "exit")
        f_int = self.fringe_integral * (
            location == "entrance"
        ) + self.fringe_integral_exit * (location == "exit")
        h_gap = 0.5 * (
            self.gap * (location == "entrance") + self.gap_exit * (location == "exit")
        )

        hx = g * jnp.tan(e)
        hy = -g * jnp.tan(
            e - 2 * f_int * h_gap * g * (1 + jnp.sin(e) ** 2) / jnp.cos(e)
        )
        px_f = px + x * hx.unsqueeze(-1)
        py_f = py + y * hy.unsqueeze(-1)

        return px_f, py_f

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        device = self.length.device
        dtype = self.length.dtype

        R_enter = self._transfer_map_enter()
        R_exit = self._transfer_map_exit()

        if jnp.any(self.length != 0.0):  # Bending magnet with finite length
            R = base_rmatrix(
                length=self.length,
                k1=self.k1,
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

        sec_e = 1.0 / jnp.cos(self._e1)
        phi = (
            self.fringe_integral
            * self.hx
            * self.gap
            * sec_e
            * (1 + jnp.sin(self._e1) ** 2)
        )

        tm = jnp.eye(7, device=device, dtype=dtype).repeat(*phi.shape, 1, 1)
        tm[..., 1, 0] = self.hx * jnp.tan(self._e1)
        tm[..., 3, 2] = -self.hx * jnp.tan(self._e1 - phi)

        return tm

    def _transfer_map_exit(self) -> jax.Array:
        """Linear transfer map for the exit face of the dipole magnet."""
        device = self.length.device
        dtype = self.length.dtype

        sec_e = 1.0 / jnp.cos(self._e2)
        phi = (
            self.fringe_integral_exit
            * self.hx
            * self.gap
            * sec_e
            * (1 + jnp.sin(self._e2) ** 2)
        )

        tm = jnp.eye(7, device=device, dtype=dtype).repeat(*phi.shape, 1, 1)
        tm[..., 1, 0] = self.hx * jnp.tan(self._e2)
        tm[..., 3, 2] = -self.hx * jnp.tan(self._e2 - phi)

        return tm

    def split(self, resolution: jnp.Array) -> list[Element]:
        # TODO: Implement splitting for dipole properly, for now just returns the
        # element itself
        return [self]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"k1={repr(self.k1)}, "
            + f"dipole_e1={repr(self.dipole_e1)},"
            + f"dipole_e2={repr(self.dipole_e2)},"
            + f"tilt={repr(self.tilt)},"
            + f"gap={repr(self.gap)},"
            + f"gap_exit={repr(self.gap_exit)},"
            + f"fringe_integral={repr(self.fringe_integral)},"
            + f"fringe_integral_exit={repr(self.fringe_integral_exit)},"
            + f"fringe_at={repr(self.fringe_at)},"
            + f"fringe_type={repr(self.fringe_type)},"
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "angle",
            "k1",
            "dipole_e1",
            "dipole_e2",
            "tilt",
            "gap",
            "gap_exit",
            "fringe_integral",
            "fringe_integral_exit",
            "fringe_at",
            "fringe_type",
            "tracking_method",
        ]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length
        plot_angle = self.angle[vector_idx] if self.angle.dim() > 0 else self.angle

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (jnp.sign(plot_angle) if self.is_active else 1)

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)
