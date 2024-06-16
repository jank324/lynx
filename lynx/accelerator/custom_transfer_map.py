from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import constants
from scipy.constants import physical_constants

from lynx.particles import Beam
from lynx.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = (
    constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge
)  # Electron mass
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class CustomTransferMap(Element):
    """
    This element can represent any custom transfer map.
    """

    def __init__(
        self,
        transfer_map: Union[jax.Array, nn.Parameter],
        length: Optional[jax.Array] = None,
        name: Optional[str] = None,
        device=None,
        dtype=jnp.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        assert isinstance(transfer_map, jax.Array)
        assert transfer_map.shape[-2:] == (7, 7)

        self._transfer_map = jnp.asarray(transfer_map, **factory_kwargs)
        self.length = (
            jnp.asarray(length, **factory_kwargs)
            if length is not None
            else jnp.zeros(transfer_map.shape[:-2], **factory_kwargs)
        )

    @classmethod
    def from_merging_elements(
        cls, elements: list[Element], incoming_beam: Beam
    ) -> "CustomTransferMap":
        """
        Combine the transfer maps of multiple successive elements into a single transfer
        map. This can be used to speed up tracking through a segment, if no changes
        are made to the elements in the segment or the energy of the beam being tracked
        through them.

        :param elements: List of consecutive elements to combine.
        :param incoming_beam: Beam entering the first element in the segment. NOTE: That
            this is required because the separate original transfer maps have to be
            computed before being combined and some of them may depend on the energy of
            the beam.
        """
        assert all(element.is_skippable for element in elements), (
            "Combining the elements in a Segment that is not skippable will result in"
            " incorrect tracking results."
        )

        device = elements[0].transfer_map(incoming_beam.energy).device
        dtype = elements[0].transfer_map(incoming_beam.energy).dtype

        tm = jnp.eye(7, device=device, dtype=dtype).repeat(
            (*incoming_beam.energy.shape, 1, 1)
        )
        for element in elements:
            tm = jnp.matmul(element.transfer_map(incoming_beam.energy), tm)
            incoming_beam = element.track(incoming_beam)

        combined_length = sum(element.length for element in elements)

        combined_name = "combined_" + "_".join(element.name for element in elements)

        return cls(
            tm, length=combined_length, device=device, dtype=dtype, name=combined_name
        )

    def transfer_map(self, energy: jax.Array) -> jax.Array:
        return self._transfer_map

    def broadcast(self, shape: Size) -> Element:
        return self.__class__(
            self._transfer_map.repeat((*shape, 1, 1)),
            length=self.length.repeat(shape),
            name=self.name,
        )

    @property
    def is_skippable(self) -> bool:
        return True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(transfer_map={repr(self._transfer_map)}, "
            + f"length={repr(self.length)}, "
            + f"name={repr(self.name)})"
        )

    def defining_features(self) -> list[str]:
        return super().defining_features + ["transfer_map"]

    def split(self, resolution: jax.Array) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        # TODO: At some point think of a nice way to indicate this in a lattice plot
        pass
