"""Utility functions for creating transfer maps for the elements."""

from typing import Optional

import jax
import jax.numpy as jnp
from scipy import constants

REST_ENERGY = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass


def rotation_matrix(angle: jax.Array) -> jax.Array:
    """Rotate the transfer map in x-y plane

    :param angle: Rotation angle in rad, for example `angle = np.pi/2` for vertical =
        dipole.
    :return: Rotation matrix to be multiplied to the element's transfer matrix.
    """
    cs = jnp.cos(angle)
    sn = jnp.sin(angle)

    tm = jnp.eye(7, dtype=angle.dtype, device=angle.device).repeat(*angle.shape, 1, 1)
    tm[..., 0, 0] = cs
    tm[..., 0, 2] = sn
    tm[..., 1, 1] = cs
    tm[..., 1, 3] = sn
    tm[..., 2, 0] = -sn
    tm[..., 2, 2] = cs
    tm[..., 3, 1] = -sn
    tm[..., 3, 3] = cs

    return tm


def base_rmatrix(
    length: jax.Array,
    k1: jax.Array,
    hx: jax.Array,
    tilt: Optional[jax.Array] = None,
    energy: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Create a universal transfer matrix for a beamline element.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param hx: Curvature (1/radius) of the element in 1/m**2.
    :param tilt: Roation of the element relative to the longitudinal axis in rad.
    :param energy: Beam energy in eV.
    :return: Transfer matrix for the element.
    """
    device = length.device
    dtype = length.dtype

    tilt = tilt if tilt is not None else jnp.zeros_like(length)
    energy = energy if energy is not None else jnp.zeros_like(length)

    gamma = energy / REST_ENERGY.to(device=device, dtype=dtype)
    igamma2 = jnp.ones_like(length)
    igamma2[gamma != 0] = 1 / gamma[gamma != 0] ** 2

    beta = jnp.sqrt(1 - igamma2)

    # Avoid division by zero
    k1 = k1.clone()
    k1[k1 == 0] = 1e-12

    kx2 = k1 + hx**2
    ky2 = -k1
    kx = jnp.sqrt(jnp.complex(kx2, 0.0))
    ky = jnp.sqrt(jnp.complex(ky2, 0.0))
    cx = jnp.cos(kx * length).real
    cy = jnp.cos(ky * length).real
    sy = jnp.clone(length)
    sy[ky != 0] = (jnp.sin(ky[ky != 0] * length[ky != 0]) / ky[ky != 0]).real

    sx = (jnp.sin(kx * length) / kx).real
    dx = hx / kx2 * (1.0 - cx)
    r56 = hx**2 * (length - sx) / kx2 / beta**2

    r56 = r56 - length / beta**2 * igamma2

    R = jnp.eye(7, dtype=dtype, device=device).repeat(*length.shape, 1, 1)
    R[..., 0, 0] = cx
    R[..., 0, 1] = sx
    R[..., 0, 5] = dx / beta
    R[..., 1, 0] = -kx2 * sx
    R[..., 1, 1] = cx
    R[..., 1, 5] = sx * hx / beta
    R[..., 2, 2] = cy
    R[..., 2, 3] = sy
    R[..., 3, 2] = -ky2 * sy
    R[..., 3, 3] = cy
    R[..., 4, 0] = sx * hx / beta
    R[..., 4, 1] = dx / beta
    R[..., 4, 5] = r56

    # Rotate the R matrix for skew / vertical magnets
    if jnp.any(tilt != 0):
        R = jnp.einsum(
            "...ij,...jk,...kl->...il", rotation_matrix(-tilt), R, rotation_matrix(tilt)
        )
    return R


def misalignment_matrix(misalignment: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Shift the beam for tracking beam through misaligned elements"""
    device = misalignment.device
    dtype = misalignment.dtype
    batch_shape = misalignment.shape[:-1]

    R_exit = jnp.eye(7, device=device, dtype=dtype).repeat(*batch_shape, 1, 1)
    R_exit[..., 0, 6] = misalignment[..., 0]
    R_exit[..., 2, 6] = misalignment[..., 1]

    R_entry = jnp.eye(7, device=device, dtype=dtype).repeat(*batch_shape, 1, 1)
    R_entry[..., 0, 6] = -misalignment[..., 0]
    R_entry[..., 2, 6] = -misalignment[..., 1]

    return R_entry, R_exit
