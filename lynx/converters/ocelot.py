import jax
import jax.numpy as jnp

import lynx


def ocelot2cheetah(
    element, warnings: bool = True, device=None, dtype=jnp.float32
) -> "lynx.Element":
    """
    Translate an Ocelot element to a Cheetah element.

    NOTE Object not supported by Cheetah are translated to drift sections. Screen
    objects are created only from `ocelot.Monitor` objects when the string "BSC" is
    contained in their `id` attribute. Their screen properties are always set to default
    values and most likely need adjusting afterwards. BPM objects are only created from
    `ocelot.Monitor` objects when their id has a substring "BPM".

    :param element: Ocelot element object representing an element of particle
        accelerator.
    :param warnings: Whether to print warnings when elements might not be converted as
        expected.
    :return: Cheetah element object representing an element of particle accelerator.
    """
    try:
        import ocelot
    except ImportError:
        raise ImportError(
            """To use the ocelot2cheetah lattice converter, Ocelot must be first
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    if isinstance(element, ocelot.Drift):
        return lynx.Drift(
            length=jnp.array([element.l], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Quadrupole):
        return lynx.Quadrupole(
            length=jnp.array([element.l], dtype=jnp.float32),
            k1=jnp.array([element.k1], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Solenoid):
        return lynx.Solenoid(
            length=jnp.array([element.l], dtype=jnp.float32),
            k=jnp.array([element.k], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Hcor):
        return lynx.HorizontalCorrector(
            length=jnp.array([element.l], dtype=jnp.float32),
            angle=jnp.array([element.angle], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Vcor):
        return lynx.VerticalCorrector(
            length=jnp.array([element.l], dtype=jnp.float32),
            angle=jnp.array([element.angle], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Bend):
        return lynx.Dipole(
            length=jnp.array([element.l], dtype=jnp.float32),
            angle=jnp.array([element.angle], dtype=jnp.float32),
            e1=jnp.array([element.e1], dtype=jnp.float32),
            e2=jnp.array([element.e2], dtype=jnp.float32),
            tilt=jnp.array([element.tilt], dtype=jnp.float32),
            fringe_integral=jnp.array([element.fint], dtype=jnp.float32),
            fringe_integral_exit=jnp.array([element.fintx], dtype=jnp.float32),
            gap=jnp.array([element.gap], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.SBend):
        return lynx.Dipole(
            length=jnp.array([element.l], dtype=jnp.float32),
            angle=jnp.array([element.angle], dtype=jnp.float32),
            e1=jnp.array([element.e1], dtype=jnp.float32),
            e2=jnp.array([element.e2], dtype=jnp.float32),
            tilt=jnp.array([element.tilt], dtype=jnp.float32),
            fringe_integral=jnp.array([element.fint], dtype=jnp.float32),
            fringe_integral_exit=jnp.array([element.fintx], dtype=jnp.float32),
            gap=jnp.array([element.gap], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.RBend):
        return lynx.RBend(
            length=jnp.array([element.l], dtype=jnp.float32),
            angle=jnp.array([element.angle], dtype=jnp.float32),
            e1=jnp.array([element.e1], dtype=jnp.float32) - element.angle / 2,
            e2=jnp.array([element.e2], dtype=jnp.float32) - element.angle / 2,
            tilt=jnp.array([element.tilt], dtype=jnp.float32),
            fringe_integral=jnp.array([element.fint], dtype=jnp.float32),
            fringe_integral_exit=jnp.array([element.fintx], dtype=jnp.float32),
            gap=jnp.array([element.gap], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.Cavity):
        return lynx.Cavity(
            length=jnp.array([element.l], dtype=jnp.float32),
            voltage=jnp.array([element.v], dtype=jnp.float32) * 1e9,
            frequency=jnp.array([element.freq], dtype=jnp.float32),
            phase=jnp.array([element.phi], dtype=jnp.float32),
            name=element.id,
            device=device,
            dtype=dtype,
        )
    elif isinstance(element, ocelot.TDCavity):
        # TODO: Better replacement at some point?
        return lynx.Cavity(
            length=jnp.array([element.l], dtype=jnp.float32),
            voltage=jnp.array([element.v], dtype=jnp.float32) * 1e9,
            frequency=jnp.array([element.freq], dtype=jnp.float32),
            phase=jnp.array([element.phi], dtype=jnp.float32),
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
            resolution=jnp.array([2448, 2040]),
            pixel_size=jnp.array([3.5488e-6, 2.5003e-6]),
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
            x_max=jnp.array([element.xmax], dtype=jnp.float32),
            y_max=jnp.array([element.ymax], dtype=jnp.float32),
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
            length=jnp.array([element.l], dtype=jnp.float32),
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
