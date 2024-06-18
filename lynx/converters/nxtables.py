import csv
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp

import lynx


def translate_element(row: list[str], header: list[str]) -> Optional[Dict]:
    """
    Translate a row of an NX Tables file to a Cheetah `Element`.

    :param row: A row of an NX Tables file as a list of column elements.
    :param header: The header row of the NX Tables file as a list of column names.
    :return: Dictionary of Cheetah `Element` object best representing the row and its
        center s position if the element is relevant for the Cheetah model, `None`
        otherwise.
    """
    class_name = row[header.index("CLASS")]
    name = row[header.index("NAME")]
    s_position = float(row[header.index("Z_beam")])

    IGNORE_CLASSES = [
        "RSBG",
        "MSOB",
        "MSOH",
        "MSOG",
        "VVAG",
        "BSCL",
        "MIRA",
        "BAML",
        "SCRL",
        "TEMG",
        "FCNG",
        "SOLE",
        "EOLE",
        "MSOL",
        "BELS",
        "VVAF",
        "MIRM",
        "SCRY",
        "FPSA",
        "VPUL",
        "SOLC",
        "SCRE",
        "SOLX",
        "ICTB",
        "BSCS",
    ]
    if class_name in IGNORE_CLASSES:
        return None
    elif class_name == "MCXG":  # TODO: Check length with Willi
        assert name[6] == "X"
        horizontal_coil = lynx.HorizontalCorrector(
            name=name[:6] + "H" + name[6 + 1 :], length=jnp.array([5e-05])
        )
        vertical_coil = lynx.VerticalCorrector(
            name=name[:6] + "V" + name[6 + 1 :], length=jnp.array([5e-05])
        )
        element = lynx.Segment(elements=[horizontal_coil, vertical_coil], name=name)
    elif class_name == "BSCX":
        element = lynx.Screen(
            name=name,
            resolution=jnp.array([2464, 2056]),
            pixel_size=jnp.array([0.00343e-3, 0.00247e-3]),
            binning=1,
        )
    elif class_name == "BSCR":
        element = lynx.Screen(
            name=name,
            resolution=jnp.array([2448, 2040]),
            pixel_size=jnp.array([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCM":
        element = lynx.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=jnp.array([2448, 2040]),
            pixel_size=jnp.array([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCO":
        element = lynx.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=jnp.array([2448, 2040]),
            pixel_size=jnp.array([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCA":
        element = lynx.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=jnp.array([2448, 2040]),
            pixel_size=jnp.array([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCE":
        element = lynx.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=jnp.array([2464, 2056]),
            pixel_size=jnp.array([0.00998e-3, 0.00715e-3]),
            binning=1,
        )
    elif class_name == "SCRD":
        element = lynx.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=jnp.array([2464, 2056]),
            pixel_size=jnp.array([0.00998e-3, 0.00715e-3]),
            binning=1,
        )
    elif class_name == "BPMG":
        element = lynx.BPM(name=name)
    elif class_name == "BPML":
        element = lynx.BPM(name=name)
    elif class_name == "SLHG":
        element = lynx.Aperture(  # TODO: Ask for actual size and shape
            name=name,
            x_max=jnp.array([jnp.inf]),
            y_max=jnp.array([jnp.inf]),
            shape="elliptical",
        )
    elif class_name == "SLHB":
        element = lynx.Aperture(  # TODO: Ask for actual size and shape
            name=name,
            x_max=jnp.array([jnp.inf]),
            y_max=jnp.array([jnp.inf]),
            shape="rectangular",
        )
    elif class_name == "SLHS":
        element = lynx.Aperture(  # TODO: Ask for actual size and shape
            name=name,
            x_max=jnp.array([jnp.inf]),
            y_max=jnp.array([jnp.inf]),
            shape="rectangular",
        )
    elif class_name == "MCHM":
        element = lynx.HorizontalCorrector(name=name, length=jnp.array([0.02]))
    elif class_name == "MCVM":
        element = lynx.VerticalCorrector(name=name, length=jnp.array([0.02]))
    elif class_name == "MBHL":
        element = lynx.Dipole(name=name, length=jnp.array([0.322]))
    elif class_name == "MBHB":
        element = lynx.Dipole(name=name, length=jnp.array([0.22]))
    elif class_name == "MBHO":
        element = lynx.Dipole(
            name=name,
            length=jnp.array([0.43852543421396856]),
            angle=jnp.array([0.8203047484373349]),
            e2=jnp.array([-0.7504915783575616]),
        )
    elif class_name == "MQZM":
        element = lynx.Quadrupole(name=name, length=jnp.array([0.122]))
    elif class_name == "RSBL":
        element = lynx.Cavity(
            name=name,
            length=jnp.array([4.139]),
            frequency=jnp.array([2.998e9]),
            voltage=jnp.array([76e6]),
        )
    elif class_name == "RXBD":
        element = lynx.Cavity(  # TODO: TD? and tilt?
            name=name,
            length=jnp.array([1.0]),
            frequency=jnp.array([11.9952e9]),
            voltage=jnp.array([0.0]),
        )
    elif class_name == "UNDA":  # TODO: Figure out actual length
        element = lynx.Undulator(name=name, length=jnp.array([0.25]))
    elif class_name in [
        "SOLG",
        "BCMG",
        "EOLG",
        "SOLS",
        "EOLS",
        "SOLA",
        "EOLA",
        "SOLT",
        "BSTB",
        "TORF",
        "EOLT",
        "SOLO",
        "EOLO",
        "SOLB",
        "EOLB",
        "ECHA",
        "MKBB",
        "MKBE",
        "MKPM",
        "EOLC",
        "SOLM",
        "EOLM",
        "SOLH",
        "BSCD",
        "STDE",  # STRIDENAS detector
        "ECHS",  # STRIDENAS chamber
        "EOLH",
        "WINA",
        "LINA",
        "EOLX",
    ]:
        element = lynx.Marker(name=name)
    else:
        raise ValueError(f"Encountered unknown class {class_name} for element {name}")

    return {"element": element, "s_position": s_position}


def read_nx_tables(filepath: Path) -> "lynx.Element":
    """
    Read an NX Tables CSV-like file generated for the ARES lattice into a Cheetah
    `Segment`.

    :param filepath: Path to the NX Tables file.
    :return: Converted Cheetah `Segment`.
    """
    with open(filepath, "r") as csvfile:
        nx_tables_rows = csv.reader(csvfile, delimiter=",")
        nx_tables_rows = list(nx_tables_rows)

    header = nx_tables_rows[0]
    nx_tables_rows = nx_tables_rows[1:]

    translated = [translate_element(row, header) for row in nx_tables_rows]
    filtered = [element for element in translated if element is not None]

    # Sort by s position
    sorted_filtered = sorted(filtered, key=lambda x: x["s_position"])

    # Insert drift sections
    filled_with_drifts = [sorted_filtered[0]["element"]]
    for previous, current in zip(sorted_filtered[:-1], sorted_filtered[1:]):
        previous_length = (
            previous["element"].length
            if hasattr(previous["element"], "length")
            else 0.0
        )
        current_length = (
            current["element"].length if hasattr(current["element"], "length") else 0.0
        )

        center_to_center_distance = current["s_position"] - previous["s_position"]
        drift_length = (
            center_to_center_distance - previous_length / 2 - current_length / 2
        )

        assert drift_length >= 0.0, (
            f"Elements {previous['element'].name} and {current['element'].name} overlap"
            f" by {drift_length}."
        )

        if drift_length > 0.0:
            filled_with_drifts.append(
                lynx.Drift(
                    name=f"DRIFT_{previous['element'].name}_{current['element'].name}",
                    length=jnp.array([drift_length]),
                )
            )

        filled_with_drifts.append(current["element"])

    segment = lynx.Segment(elements=filled_with_drifts, name=filepath.stem)

    # Return flattened because conversion prduces nested segments
    return segment.flattened()
