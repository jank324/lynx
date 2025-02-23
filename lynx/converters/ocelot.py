import jax
import jax.numpy as jnp

import lynx


def convert_element_to_lynx(
    element, warnings: bool = True, device=None, dtype=jnp.float32
) -> "lynx.Element":
    """
    Translate an Ocelot element to a Lynx element.

    NOTE Object not supported by Lynx are translated to drift sections. Screen
    objects are created only from `ocelot.Monitor` objects when the string "BSC" is
    contained in their `id` attribute. Their screen properties are always set to default
    values and most likely need adjusting afterwards. BPM objects are only created from
    `ocelot.Monitor` objects when their id has a substring "BPM".

    :param element: Ocelot element object representing an element of particle
        accelerator.
    :param warnings: Whether to print warnings when elements might not be converted as
        expected.
    :return: Lynx element object representing an element of particle accelerator.
    """
    try:
        import ocelot
    except ImportError:
        raise ImportError(
            """To use the ocelot2lynx lattice converter, Ocelot must be first
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    if isinstance(element, ocelot.Drift):
        return lynx.Drift(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Quadrupole):
        return lynx.Quadrupole(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            k1=jnp.asarray(element.k1, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Solenoid):
        return lynx.Solenoid(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            k=jnp.asarray(element.k, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Hcor):
        return lynx.HorizontalCorrector(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            angle=jnp.asarray(element.angle, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Vcor):
        return lynx.VerticalCorrector(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            angle=jnp.asarray(element.angle, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Bend):
        return lynx.Dipole(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            angle=jnp.asarray(element.angle, dtype=jnp.float32),
            dipole_e1=jnp.asarray(element.e1, dtype=jnp.float32),
            dipole_e2=jnp.asarray(element.e2, dtype=jnp.float32),
            tilt=jnp.asarray(element.tilt, dtype=jnp.float32),
            fringe_integral=jnp.asarray(element.fint, dtype=jnp.float32),
            fringe_integral_exit=jnp.asarray(element.fintx, dtype=jnp.float32),
            gap=jnp.asarray(element.gap, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.SBend):
        return lynx.Dipole(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            angle=jnp.asarray(element.angle, dtype=jnp.float32),
            dipole_e1=jnp.asarray(element.e1, dtype=jnp.float32),
            dipole_e2=jnp.asarray(element.e2, dtype=jnp.float32),
            tilt=jnp.asarray(element.tilt, dtype=jnp.float32),
            fringe_integral=jnp.asarray(element.fint, dtype=jnp.float32),
            fringe_integral_exit=jnp.asarray(element.fintx, dtype=jnp.float32),
            gap=jnp.asarray(element.gap, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.RBend):
        return lynx.RBend(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            angle=jnp.asarray(element.angle, dtype=jnp.float32),
            rbend_e1=jnp.asarray(element.e1, dtype=jnp.float32) - element.angle / 2,
            rbend_e2=jnp.asarray(element.e2, dtype=jnp.float32) - element.angle / 2,
            tilt=jnp.asarray(element.tilt, dtype=jnp.float32),
            fringe_integral=jnp.asarray(element.fint, dtype=jnp.float32),
            fringe_integral_exit=jnp.asarray(element.fintx, dtype=jnp.float32),
            gap=jnp.asarray(element.gap, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Cavity):
        return lynx.Cavity(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            voltage=jnp.asarray(element.v, dtype=jnp.float32) * 1e9,
            frequency=jnp.asarray(element.freq, dtype=jnp.float32),
            phase=jnp.asarray(element.phi, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.TDCavity):
        # TODO: Better replacement at some point?
        return lynx.Cavity(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            voltage=jnp.asarray(element.v, dtype=jnp.float32) * 1e9,
            frequency=jnp.asarray(element.freq, dtype=jnp.float32),
            phase=jnp.asarray(element.phi, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Monitor) and ("BSC" in element.id):
        # NOTE This pattern is very specific to ARES and will need a more complex
        # solution for other accelerators
        if warnings:
            print(
                "WARNING: Diagnostic screen was converted with default screen"
                " properties."
            )
        return lynx.Screen(
            resolution=(2448, 2040),
            pixel_size=jnp.asarray([3.5488e-6, 2.5003e-6]),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Monitor) and "BPM" in element.id:
        return lynx.BPM(name=element.id)
    elif isinstance(element, ocelot.Marker):
        return lynx.Marker(name=element.id)
    elif isinstance(element, ocelot.Monitor):
        return lynx.Marker(name=element.id)
    elif isinstance(element, ocelot.Undulator):
        return lynx.Undulator(
            length=jnp.array([element.l], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Aperture):
        shape_translation = {"rect": "rectangular", "elip": "elliptical"}
        return lynx.Aperture(
            x_max=jnp.asarray(element.xmax, dtype=jnp.float32),
            y_max=jnp.asarray(element.ymax, dtype=jnp.float32),
            shape=shape_translation[element.type],
            is_active=True,
            name=element.id,
            device=device,
            dtype=dtype,
        )
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id} of type {type(element)},"
                " replacing with drift section."
            )
        return lynx.Drift(
            length=jnp.asarray(element.l, dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )


def subcell_of_ocelot(cell: list, start: str, end: str) -> list:
    """Extract a subcell `[start, end]` from an Ocelot cell."""
    subcell = []
    is_in_subcell = False
    for el in cell:
        if el.id == start:
            is_in_subcell = True
        if is_in_subcell:
            subcell.append(el)
        if el.id == end:
            break

    return subcell
