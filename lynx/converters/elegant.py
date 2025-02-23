from pathlib import Path
from typing import Optional, Union

import jax.numpy as jnp

import lynx
from lynx.converters.utils.fortran_namelist import (
    merge_delimiter_continued_lines,
    parse_lines,
    read_clean_lines,
    validate_understood_properties,
)


def convert_element(
    name: str,
    context: dict,
    device: Optional[Union[str, jnp.device]] = None,
    dtype: jnp.dtype = jnp.float32,
) -> "lynx.Element":
    """Convert a parsed elegant element dict to a lynx Element.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from elegant lattice file(s).
    :param device: Device to put the element on. If `None`, the device is set to
        `jnp.device("cpu")`.
    :param dtype: Data type to use for the element. Default is `jnp.float32`.
    :return: Converted lynx Element. If you are calling this function yourself
        as a user of Lynx, this is most likely a `Segment`.
    """
    parsed = context[name]

    if isinstance(parsed, list):
        return lynx.Segment(
            elements=[
                convert_element(element_name, context, device, dtype)
                for element_name in parsed
            ],
            name=name,
        )
    elif isinstance(parsed, dict) and "element_type" in parsed:
        if parsed["element_type"] == "sole":
            # The group property does not have an analoge in Lynx, so it is neglected
            validate_understood_properties(["element_type", "l", "group"], parsed)
            return lynx.Solenoid(
                length=jnp.asarray(parsed["l"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["hkick", "hkic"]:
            validate_understood_properties(
                ["element_type", "l", "kick", "group"], parsed
            )
            return lynx.HorizontalCorrector(
                length=jnp.asarray(parsed.get("l", 0.0)),
                angle=jnp.asarray(parsed.get("kick", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["vkick", "vkic"]:
            validate_understood_properties(
                ["element_type", "l", "kick", "group"], parsed
            )
            return lynx.VerticalCorrector(
                length=jnp.asarray(parsed.get("l", 0.0)),
                angle=jnp.asarray(parsed.get("kick", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["mark", "marker"]:
            validate_understood_properties(["element_type", "group"], parsed)
            return lynx.Marker(name=name)
        elif parsed["element_type"] == "kick":
            validate_understood_properties(["element_type", "l", "group"], parsed)

            # TODO Find proper element class
            return lynx.Drift(
                length=jnp.asarray(parsed.get("l", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["drift", "drif"]:
            validate_understood_properties(["element_type", "l", "group"], parsed)
            return lynx.Drift(
                length=jnp.asarray(parsed.get("l", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["csrdrift", "csrdrif"]:
            # Drift that includes effects from coherent synchrotron radiation
            validate_understood_properties(
                ["element_type", "l", "group", "use_stupakov", "n_kicks", "csr"], parsed
            )
            return lynx.Drift(
                length=jnp.asarray(parsed.get("l", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["lscdrift", "lscdrif"]:
            # Drift that includes space charge effects
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "group",
                    "interpolate",
                    "smoothing",
                    "bins",
                    "high_frequency_cutoff0",
                    "high_frequency_cutoff1",
                    "lsc",
                ],
                parsed,
            )
            return lynx.Drift(
                length=jnp.asarray(parsed.get("l", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "ecol":
            validate_understood_properties(
                ["element_type", "l", "x_max", "y_max"],
                parsed,
            )
            return lynx.Segment(
                elements=[
                    lynx.Drift(
                        length=jnp.asarray(parsed.get("l", 0.0)),
                        name=name + "_drift",
                        device=device,
                        dtype=dtype,
                    ),
                    lynx.Aperture(
                        x_max=jnp.asarray(parsed.get("x_max", jnp.inf)),
                        y_max=jnp.asarray(parsed.get("y_max", jnp.inf)),
                        shape="elliptical",
                        name=name + "_aperture",
                        device=device,
                        dtype=dtype,
                    ),
                ],
                name=name + "_segment",
            )
        elif parsed["element_type"] == "rcol":
            validate_understood_properties(
                ["element_type", "l", "x_max", "y_max"],
                parsed,
            )
            return lynx.Segment(
                elements=[
                    lynx.Drift(
                        length=jnp.asarray(parsed.get("l", 0.0)),
                        name=name + "_drift",
                        device=device,
                        dtype=dtype,
                    ),
                    lynx.Aperture(
                        x_max=jnp.asarray(parsed.get("x_max", jnp.inf)),
                        y_max=jnp.asarray(parsed.get("y_max", jnp.inf)),
                        shape="rectangular",
                        name=name + "_aperture",
                        device=device,
                        dtype=dtype,
                    ),
                ],
                name=name + "_segment",
            )
        elif parsed["element_type"] in ["quad", "quadrupole"]:
            validate_understood_properties(
                ["element_type", "l", "k1", "tilt", "group"],
                parsed,
            )
            return lynx.Quadrupole(
                length=jnp.asarray(parsed["l"]),
                k1=jnp.asarray(parsed["k1"]),
                tilt=jnp.asarray(parsed.get("tilt", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "sext":
            # validate_understood_properties(
            #     ["element_type", "l", "group"],
            #     parsed,
            # )

            # TODO Parse properly! Missing element class
            return lynx.Drift(
                length=jnp.asarray(parsed["l"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "moni":
            validate_understood_properties(["element_type", "group", "l"], parsed)
            if "l" in parsed:
                return lynx.Segment(
                    elements=[
                        lynx.Drift(
                            length=jnp.asarray(parsed["l"] / 2),
                            name=name + "_predrift",
                            device=device,
                            dtype=dtype,
                        ),
                        lynx.BPM(name=name),
                        lynx.Drift(
                            length=jnp.asarray(parsed["l"] / 2),
                            name=name + "_postdrift",
                            device=device,
                            dtype=dtype,
                        ),
                    ],
                    name=name + "_segment",
                )
            else:
                return lynx.BPM(name=name)
        elif parsed["element_type"] == "ematrix":
            validate_understood_properties(
                ["element_type", "l", "order", "c[1-6]", "r[1-6][1-6]", "group"],
                parsed,
            )

            if parsed.get("order", 1) != 1:
                raise ValueError("Only first order modelling is supported")

            # Initially zero in elegant by convention
            R = jnp.zeros((7, 7), device=device, dtype=dtype)
            # Add linear component
            R[:6, :6] = jnp.asarray(
                [
                    [parsed.get(f"r{i + 1}{j + 1}", 0.0) for j in range(6)]
                    for i in range(6)
                ],
                device=device,
                dtype=dtype,
            )
            # Add affine component (constant offset)
            R[:6, 6] = jnp.asarray(
                [parsed.get(f"c{i + 1}", 0.0) for i in range(6)],
                device=device,
                dtype=dtype,
            )

            return lynx.CustomTransferMap(
                length=jnp.asarray(parsed["l"]),
                predefined_transfer_map=R,
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "rfca":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "phase",
                    "volt",
                    "freq",
                    "change_p0",
                    "end1_focus",
                    "end2_focus",
                    "body_focus_model",
                    "group",
                ],
                parsed,
            )

            # TODO Properly handle all parameters
            return lynx.Cavity(
                length=jnp.asarray(parsed["l"]),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Lynx uses 0°. We therefore add a phase offset to compensate.
                phase=jnp.asarray(parsed["phase"] - 90),
                voltage=jnp.asarray(parsed["volt"]),
                frequency=jnp.asarray(parsed["freq"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "rfcw":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "phase",
                    "volt",
                    "freq",
                    "change_p0",
                    "end1_focus",
                    "end2_focus",
                    "cell_length",
                    "zwakefile",
                    "trwakefile",
                    "tcolumn",
                    "wxcolumn",
                    "wycolumn",
                    "wzcolumn",
                    "interpolate",
                    "n_kicks",
                    "smoothing",
                    "zwake",
                    "trwake",
                    "lsc",
                    "lsc_bins",
                    "lsc_high_frequency_cutoff0",
                    "lsc_high_frequency_cutoff1",
                    "group",
                ],
                parsed,
            )

            # TODO Properly handle all parameters
            return lynx.Cavity(
                length=jnp.asarray(parsed["l"]),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Lynx uses 0°. We therefore add a phase offset to compensate.
                phase=jnp.asarray(parsed["phase"] - 90),
                voltage=jnp.asarray(parsed["volt"]),
                frequency=jnp.asarray(parsed["freq"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "rfdf":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "phase",
                    "voltage",
                    "frequency",
                    "group",
                ],
                parsed,
            )

            # TODO Properly handle all parameters
            return lynx.TransverseDeflectingCavity(
                length=jnp.asarray(parsed["l"]),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Lynx uses 0°. We therefore add a phase offset to compensate.
                phase=jnp.asarray(parsed["phase"] - 90),
                voltage=jnp.asarray(parsed["voltage"]),
                frequency=jnp.asarray(parsed["frequency"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] in ["sben", "csbend"]:
            validate_understood_properties(
                ["element_type", "l", "angle", "k1", "e1", "e2", "tilt", "group"],
                parsed,
            )
            return lynx.Dipole(
                length=jnp.asarray(parsed["l"]),
                angle=jnp.asarray(parsed.get("angle", 0.0)),
                k1=jnp.asarray(parsed.get("k1", 0.0)),
                dipole_e1=jnp.asarray(parsed.get("e1", 0.0)),
                dipole_e2=jnp.asarray(parsed.get("e2", 0.0)),
                tilt=jnp.asarray(parsed.get("tilt", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "rben":
            validate_understood_properties(
                ["element_type", "l", "angle", "e1", "e2", "tilt", "group"],
                parsed,
            )
            return lynx.RBend(
                length=jnp.asarray(parsed["l"]),
                angle=jnp.asarray(parsed.get("angle", 0.0)),
                rbend_e1=jnp.asarray(parsed.get("e1", 0.0)),
                rbend_e2=jnp.asarray(parsed.get("e2", 0.0)),
                tilt=jnp.asarray(parsed.get("tilt", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "csrcsben":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "angle",
                    "e1",
                    "e2",
                    "edge1_effects",
                    "edge2_effects",
                    "tilt",
                    "hgap",
                    "fint",
                    "sg_halfwidth",
                    "sg_order",
                    "steady_state",
                    "bins",
                    "n_kicks",
                    "integration_order",
                    "isr",
                    "csr",
                    "group",
                ],
                parsed,
            )
            return lynx.Dipole(
                length=jnp.asarray(parsed["l"]),
                angle=jnp.asarray(parsed.get("angle", 0.0)),
                k1=jnp.asarray(parsed.get("k1", 0.0)),
                dipole_e1=jnp.asarray(parsed.get("e1", 0.0)),
                dipole_e2=jnp.asarray(parsed.get("e2", 0.0)),
                tilt=jnp.asarray(parsed.get("tilt", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif parsed["element_type"] == "watch":
            validate_understood_properties(
                ["element_type", "group", "filename"], parsed
            )
            return lynx.Marker(name=name)
        elif parsed["element_type"] in ["charge", "wake"]:
            print(
                f"WARNING: Information provided in element {name} of type"
                f" {parsed['element_type']} cannot be imported automatically. Consider"
                " manually providing the correct information."
            )
            return lynx.Marker(name=name)
        else:
            print(
                f"WARNING: Element {name} of type {parsed['element_type']} cannot"
                " be converted correctly. Using drift section instead."
            )
            # TODO: Remove the length if by adding markers to Lynx
            return lynx.Drift(
                name=name,
                length=jnp.asarray(parsed.get("l", 0.0)),
                device=device,
                dtype=dtype,
            )
    else:
        raise ValueError(
            f"Unknown elegant element type for {name = }"  # noqa: E202, E251
        )


def convert_lattice_to_lynx(
    elegant_lattice_file_path: Path,
    name: str,
    device: Optional[Union[str, jnp.device]] = None,
    dtype: jnp.dtype = jnp.float32,
) -> "lynx.Element":
    """
    Convert a elegant lattice file to a Lynx `Segment`.

    :param elegant_lattice_file_path: Path to the elegant lattice file.
    :param name: Name of the root element.
    :param device: Device to use for the lattice. If `None`, the device is set to
        `jnp.device("cpu")`.
    :param dtype: Data type to use for the lattice. Default is `jnp.float32`.
    :return: Lynx `Segment` representing the elegant lattice.
    """

    # Read and clean the lattice file(s)
    lines = read_clean_lines(elegant_lattice_file_path)

    # Merge multi-line statements
    merged_lines = merge_delimiter_continued_lines(
        lines, delimiter="&", remove_delimiter=True
    )
    merged_lines = merge_delimiter_continued_lines(
        merged_lines, delimiter=",", remove_delimiter=False
    )
    merged_lines = merge_delimiter_continued_lines(
        merged_lines, delimiter="{", remove_delimiter=False
    )
    assert len(merged_lines) <= len(
        lines
    ), "Merging lines should never produce more lines than there were before."

    # Parse the lattice file(s), i.e. basically execute them
    context = parse_lines(merged_lines)

    # Convert the parsed lattice info to Lynx elements
    return convert_element(name, context, device, dtype)
