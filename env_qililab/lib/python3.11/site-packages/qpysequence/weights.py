# Copyright 2023 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bisect import insort
from dataclasses import dataclass

import numpy as np


@dataclass
class Weight:
    """Weight dataclass"""

    index: int
    name: str
    data: list[float] | np.ndarray


@dataclass
class WeightPair:
    """WeightPair dataclass"""

    weight_i: Weight
    weight_q: Weight
    name: str


class Weights:
    """Weights class. Holds all the weights used in a sequence."""

    def __init__(self):
        self._used_indices: list[int] = []
        self._weights: list[Weight] = []
        self._weight_pairs: list[WeightPair] = []

    def _find_available_index(self) -> int:
        """Finds the smallest available index

        Returns:
            index (int): available index.
        """
        index = -1
        for proposed, existing in enumerate(self._used_indices):
            if proposed != existing:
                index = proposed
        if index == -1:
            index = len(self._used_indices)
        return index

    def add(self, array: list[float] | np.ndarray, index: int = -1, name: str | None = None) -> int:
        """Adds a weight to the Weights object.

        Args:
            array (list[float] | np.ndarray): array containing the weight.
            index (int, optional): index of the weight. Defaults to None.
            name (str, optional): string identifier of the weight. Defaults to -1.

        Returns:
            index (int): index of the weight.
        """
        # Check if the index is valid or find one if not given.
        if index == -1:
            index = self._find_available_index()
        elif index in self._used_indices:
            raise IndexError(f"Index {index} is already used.")
        # Add the index to the used_indices list
        insort(self._used_indices, index)
        # Auto-name if name not given.
        if not name:
            name = f"weight_{index}"
        # If array is a numpy array, convert to list
        if isinstance(array, np.ndarray):
            array = array.tolist()  # type: ignore[assignment]
        # Create weight object and add to weights list
        weight = Weight(index, name, array)
        self._weights.append(weight)
        return index

    def add_pair(
        self,
        pair: tuple[list[float] | np.ndarray, list[float] | np.ndarray],
        indices: tuple[int, int] = (-1, -1),
        name: str | None = None,
    ) -> tuple[int, int]:
        """Adds a weight pair to the Weights object.

        Args:
            pair (tuple[list[float] | np.ndarray, list[float] | np.ndarray]): tuple with two arrays containing two
                weights.
            indices (tuple(int, int), optional): tuple with the indices of the weights. Defaults to None.
            name (str, optional): string identifier of the weight. Defaults to None.

        Returns:
            indices (tuple[int, int]): indices of the weights.
        """
        # Auto-name if name not given
        if not name:
            name = f"pair_{len(self._weight_pairs)}"
        i_index, q_index = indices
        i_index = self.add(pair[0], i_index, f"{name}_I")
        q_index = self.add(pair[1], q_index, f"{name}_Q")
        return self._create_weight_pair(name, i_index, q_index)

    def add_pair_from_complex(
        self,
        array: np.ndarray,
        indices: tuple[int, int] = (-1, -1),
        name: str | None = None,
    ) -> tuple[int, int]:
        """Adds the weights of real and imaginary part waveforms as a weight pair to the Weights object.

        Args:
            array (np.ndarray): complex numpy array.
            indices (tuple(int, int), optional): tuple with the indices of the waveforms. Defaults to None.
            name (str, optional): string identifier of the waveform. Defaults to None.

        Returns:
            indices (tuple[int, int]): indices of the waveforms.
        """
        # Auto-name if name not given
        if not name:
            name = f"pair_{len(self._weight_pairs)}"
        i_index, q_index = indices
        i_index = self.add(array.real, i_index, f"{name}_I")
        q_index = self.add(array.imag, q_index, f"{name}_Q")
        return self._create_weight_pair(name, i_index, q_index)

    def _create_weight_pair(self, name: str, i_index: int, q_index: int):
        """Creates a waveform pair.

        Args:
            name (str): name of the waveform pair.
            i_index (int): index of the I component.
            q_index (int): index of the Q component.
        """
        weights_pair = WeightPair(self._weights[-2], self._weights[-1], name)
        self._weight_pairs.append(weights_pair)
        return i_index, q_index

    def find_by_name(self, name: str) -> Weight:
        """Finds a Weight by name

        Args:
            name (str)): Name of the Weight

        Raises:
            NameError: Weight not found

        Returns:
            Weight: Weight found by name.
        """
        for weight in self._weights:
            if weight.name == name:
                return weight
        raise NameError(f"Could not find Weight with name {name}.")

    def find_pair_by_name(self, name: str) -> WeightPair:
        """Finds a Weight pair by name.

        Args:
            name (str): Name of the weight pair.

        Raises:
            NameError: Weight pair not found.

        Returns:
            WeightPair: Weight pair found by name.
        """
        for pair in self._weight_pairs:
            if pair.name == name:
                return pair
        raise NameError(f"Could not find Weight pair with name {name}.")

    def to_dict(self) -> dict:
        """Creates a dictionary representation of the Weights object. It can be passed directly to a Qblox sequencer
        when converted to string.

        Returns:
            dict (dict): dictionary representation of the weights.
        """
        dictionary = {}
        for weight in self._weights:
            entry = {"data": weight.data, "index": weight.index}
            dictionary[weight.name] = entry
        return dictionary

    def __repr__(self) -> str:
        """JSON string representation of the Weights obejct. It can be passed directly to a Qblox sequencer.

        Returns:
            str: JSON representation of the weights."""
        return str(self.to_dict())
